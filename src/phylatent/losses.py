"""PhyLatent auxiliary losses; original detach and sampling semantics preserved."""
import torch
import torch.nn.functional as F

def normalized_mse(a, b):
    return (F.normalize(a.float(), dim=-1) - F.normalize(b.float(), dim=-1)).pow(2).mean()

def apply_static_visual_jitter(pixels, cfg):
    x = pixels.clone()
    b = float(cfg.loss.invariance.brightness)
    c = float(cfg.loss.invariance.channel_jitter)
    brightness = torch.empty(x.size(0), x.size(1), 1, 1, 1, device=x.device).uniform_(-b, b)
    channels = torch.empty(x.size(0), x.size(1), x.size(2), 1, 1, device=x.device).uniform_(-c, c)
    return x + brightness + channels

def action_separation_loss(pred_emb, pred_cf, ctx_action, cf_action, cfg):
    action_delta = (cf_action - ctx_action).float().norm(dim=-1)
    action_delta = action_delta / (ctx_action.size(-1) ** 0.5)
    pred_gap = (pred_cf - pred_emb.detach()).float().norm(dim=-1)
    pred_gap = pred_gap / (pred_cf.size(-1) ** 0.5)
    target_gap = float(cfg.loss.action_separation.margin_scale) * action_delta
    target_gap = torch.clamp(target_gap, max=float(cfg.loss.action_separation.max_gap))
    active = action_delta > action_delta.median().detach()
    if active.any():
        return F.relu(target_gap[active] - pred_gap[active]).mean()
    return F.relu(target_gap - pred_gap).mean()

def diffusion_aux_loss(self, pred_emb, tgt_emb, ctx_act_emb, cfg):
    sigma_min = float(cfg.loss.diffusion.noise_min)
    sigma_max = float(cfg.loss.diffusion.noise_max)
    sigma = torch.empty(*tgt_emb.shape[:-1], 1, device=tgt_emb.device).uniform_(sigma_min, sigma_max)
    noise = torch.randn_like(tgt_emb.float())
    noisy = tgt_emb.float() + sigma * noise
    action_cond = ctx_act_emb[:, : pred_emb.size(1)].float()
    noise_pred = self.latent_denoiser(noisy, pred_emb.float(), action_cond, sigma)
    return F.mse_loss(noise_pred, noise)
