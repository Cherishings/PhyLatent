"""Fresh PhyLatent training; no fine-tuning, recovery, or checkpoint selection."""
import os
from functools import partial
from pathlib import Path
import torch
import torch.nn.functional as F
from .models.heads import StateHead, ProjectionHead, ActionQueryAlign, LatentDenoiser
from .models.modules import SIGReg
from .losses import normalized_mse, apply_static_visual_jitter, action_separation_loss, diffusion_aux_loss

def phylatent_forward(self, batch, stage, cfg):
    """LeWM plus dynamics-relevant latent constraints."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = self.sigreg(emb.transpose(0, 1))

    state_key = cfg.loss.state.key
    state_loss = torch.zeros((), device=emb.device)
    if state_key in batch:
        state = torch.nan_to_num(batch[state_key].float(), 0.0)
        state_cur = self.state_head(emb.float())
        state_pred = self.state_head(pred_emb.float())
        state_loss = F.mse_loss(state_cur, state) + F.mse_loss(state_pred, state[:, n_preds:])

    pred_align = self.future_align(pred_emb.float())
    true_align = self.future_align(tgt_emb.float()).detach()
    align_loss = normalized_mse(pred_align, true_align)

    pred_attn = self.action_query(pred_emb.float(), ctx_act.float())
    true_attn = self.action_query(tgt_emb.float(), ctx_act.float()).detach()
    attn_align_loss = normalized_mse(pred_attn, true_attn)

    invariance_loss = torch.zeros((), device=emb.device)
    if float(cfg.loss.invariance.weight) > 0:
        aug_batch = dict(batch)
        aug_batch["pixels"] = apply_static_visual_jitter(batch["pixels"], cfg)
        aug_output = self.model.encode(aug_batch)
        invariance_loss = normalized_mse(aug_output["emb"].float(), emb.detach().float())

    sep_loss = torch.zeros((), device=emb.device)
    if float(cfg.loss.action_separation.weight) > 0 and batch["action"].size(0) > 1:
        perm = torch.randperm(batch["action"].size(0), device=batch["action"].device)
        cf_action = batch["action"][:, :ctx_len].clone()
        cf_action = batch["action"][perm, :ctx_len]
        noise_std = float(cfg.loss.action_separation.noise_std)
        cf_action = cf_action + noise_std * torch.randn_like(cf_action)
        cf_act_emb = self.model.action_encoder(cf_action)
        pred_cf = self.model.predict(ctx_emb, cf_act_emb)
        sep_loss = action_separation_loss(pred_emb, pred_cf, batch["action"][:, :ctx_len], cf_action, cfg)

    diffusion_loss = torch.zeros((), device=emb.device)
    if float(cfg.loss.diffusion.weight) > 0:
        diffusion_loss = diffusion_aux_loss(self, pred_emb, tgt_emb.detach(), ctx_act, cfg)

    output["pred_loss"] = pred_loss
    output["sigreg_loss"] = sigreg_loss
    output["state_loss"] = state_loss
    output["align_loss"] = align_loss + attn_align_loss
    output["invariance_loss"] = invariance_loss
    output["action_separation_loss"] = sep_loss
    output["diffusion_loss"] = diffusion_loss
    output["loss"] = (
        pred_loss
        + float(cfg.loss.sigreg.weight) * sigreg_loss
        + float(cfg.loss.state.weight) * state_loss
        + float(cfg.loss.align.weight) * output["align_loss"]
        + float(cfg.loss.invariance.weight) * invariance_loss
        + float(cfg.loss.action_separation.weight) * sep_loss
        + float(cfg.loss.diffusion.weight) * diffusion_loss
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

def train(cfg):
    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from omegaconf import OmegaConf, open_dict
    from .data import get_column_normalizer, get_img_preprocessor
    from .checkpointing import ExportModel
    pl.seed_everything(int(cfg.seed), workers=True)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    world_model = hydra.utils.instantiate(cfg.model)

    state_dim = dataset.get_dim(cfg.loss.state.key)
    latent_dim = cfg.embed_dim
    state_head = StateHead(latent_dim, cfg.method.state_hidden_dim, state_dim)
    future_align = ProjectionHead(latent_dim, cfg.method.align_dim)
    action_query = ActionQueryAlign(latent_dim, cfg.method.attention_heads)
    latent_denoiser = LatentDenoiser(latent_dim, cfg.method.denoise_hidden_dim)

    optimizers = {
        "model_opt": {
            "modules": "model|state_head|future_align|action_query|latent_denoiser",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 1},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        state_head=state_head,
        future_align=future_align,
        action_query=action_query,
        latent_denoiser=latent_denoiser,
        forward=partial(phylatent_forward, cfg=cfg),
        optim=optimizers,
    )

    run_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Fresh training requires an empty output directory: {run_dir}")

    logger = None
    if cfg.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ExportModel(run_dir / "model", cfg.model)

    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=str(run_dir),
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(trainer=trainer, module=world_model, data=data_module,
        ckpt_path=None)

    manager()


def apply_ablation(cfg):
    """Disable complete auxiliary objectives without changing initialization."""
    from omegaconf import open_dict
    valid = {"state", "align", "invariance", "action_separation", "diffusion"}
    disabled = list(cfg.ablation.disable)
    if not set(disabled) <= valid:
        raise ValueError(f"Unknown ablation component: {set(disabled) - valid}")
    with open_dict(cfg):
        for component in disabled:
            cfg.loss[component].weight = 0.0
    return cfg
