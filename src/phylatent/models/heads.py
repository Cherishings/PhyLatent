"""Training-only auxiliary heads for PhyLatent."""
import torch
from torch import nn

class StateHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, z):
        return self.net(z)

class ProjectionHead(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)

class ActionQueryAlign(nn.Module):
    """Action-conditioned current/future attention alignment head."""

    def __init__(self, latent_dim: int, heads: int):
        super().__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latents: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        query = self.query(action_emb.mean(dim=1, keepdim=True))
        key = self.key(latents)
        value = self.value(latents)
        out, _ = self.attn(query, key, value, need_weights=False)
        return self.norm(out.squeeze(1))

class LatentDenoiser(nn.Module):
    """Diffusion-style denoising auxiliary head for future latent prediction."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 3 + 1),
            nn.Linear(latent_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, noisy_target, pred_cond, action_cond, sigma):
        sigma_feat = sigma.expand(*noisy_target.shape[:-1], 1)
        x = torch.cat([noisy_target, pred_cond, action_cond, sigma_feat], dim=-1)
        return self.net(x)
