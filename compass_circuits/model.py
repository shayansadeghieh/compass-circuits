"""Minimal causal transformer, with explicit attention and named hook sites.

Use PyTorch forward hooks on block.attn_out_hook, block.mlp_out_hook,
block.resid_pre_hook, or block.resid_post_hook. A hook may return a
replacement tensor, enabling activation patching.
"""

from dataclasses import dataclass, asdict
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    seq_len: int = 12
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 128


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.attn_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model))
        self.resid_pre_hook = nn.Identity()
        self.attn_out_hook = nn.Identity()
        self.mlp_out_hook = nn.Identity()
        self.resid_post_hook = nn.Identity()

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        resid = self.resid_pre_hook(resid)
        b, t, d = resid.shape
        qkv = self.qkv(self.ln1(resid)).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # B,H,T,Dh
        scores = (q @ k.transpose(-2, -1)) / sqrt(self.d_head)
        causal_mask = torch.ones(t, t, dtype=torch.bool, device=resid.device).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        values = (attn @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        attn_out = self.attn_out_hook(self.attn_proj(values))
        resid = resid + attn_out
        mlp_out = self.mlp_out_hook(self.mlp(self.ln2(resid)))
        return self.resid_post_hook(resid + mlp_out)


class CompassTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(4, cfg.d_model)
        self.positions = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.head_x = nn.Linear(cfg.d_model, 2 * cfg.seq_len + 1)
        self.head_y = nn.Linear(cfg.d_model, 2 * cfg.seq_len + 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = tokens.shape
        if t != self.cfg.seq_len:
            raise ValueError(f"Expected {self.cfg.seq_len} tokens; got {t}")
        resid = self.tokens(tokens) + self.positions(torch.arange(t, device=tokens.device))[None, :, :]
        for block in self.blocks:
            resid = block(resid)
        final = self.final_ln(resid[:, -1, :])
        return self.head_x(final), self.head_y(final)


def loss_and_accuracy(logits: tuple[torch.Tensor, torch.Tensor], xy: torch.Tensor, seq_len: int):
    lx, ly = logits
    labels = xy + seq_len  # map [-seq_len,+seq_len] to [0,2*seq_len]
    loss = (F.cross_entropy(lx, labels[:, 0]) + F.cross_entropy(ly, labels[:, 1])) / 2
    prediction = torch.stack([lx.argmax(-1), ly.argmax(-1)], dim=-1) - seq_len
    joint = (prediction == xy).all(dim=-1).float().mean()
    per_axis = (prediction == xy).float().mean(dim=0)
    return loss, joint, per_axis


def hook_site(model: CompassTransformer, layer: int, name: str) -> nn.Module:
    if not 0 <= layer < len(model.blocks):
        raise ValueError("Invalid layer index")
    if name not in ("resid_pre", "attn_out", "mlp_out", "resid_post"):
        raise ValueError("Unknown hook site")
    return getattr(model.blocks[layer], name + "_hook")


def load_model(path: str, device: torch.device) -> CompassTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CompassTransformer(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def save_model(model: CompassTransformer, path: str) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)
