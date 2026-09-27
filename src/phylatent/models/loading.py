"""Build a ViT-Tiny/14 JEPA and load exported state dictionaries offline."""
import json
from pathlib import Path
import torch
from torch import nn
from transformers import ViTConfig, ViTModel
from .jepa import JEPA
from .modules import ARPredictor, Embedder, MLP


def build_encoder(size="tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False):
    if size != "tiny" or pretrained:
        raise ValueError("This release supports randomly initialized ViT-Tiny only.")
    config = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
        intermediate_size=768, image_size=image_size, patch_size=patch_size)
    return ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)


def build_model(config):
    clean = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    def mlp(key):
        c = clean(config[key]); c.pop("norm_fn", None)
        return MLP(**c, norm_fn=nn.BatchNorm1d)
    return JEPA(encoder=build_encoder(**clean(config["encoder"])),
        predictor=ARPredictor(**clean(config["predictor"])),
        action_encoder=Embedder(**clean(config["action_encoder"])),
        projector=mlp("projector"), pred_proj=mlp("pred_proj"))


def load_model(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model = build_model(config)
    state = torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
