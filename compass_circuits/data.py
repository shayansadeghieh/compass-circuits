"""Synthetic N/E/S/W sequences and deterministic displacement targets."""

import torch

TOKENS = ("N", "E", "S", "W")
# N=(0,+1), E=(+1,0), S=(0,-1), W=(-1,0)
DXY = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=torch.long)


def coordinates(tokens: torch.Tensor) -> torch.Tensor:
    """Return displacement for each prefix, shape [batch, seq, 2]."""
    # Integer comparisons work on CPU, MPS, and CUDA without device transfers.
    x = (tokens == 1).long() - (tokens == 3).long()
    y = (tokens == 0).long() - (tokens == 2).long()
    return torch.stack([x.cumsum(dim=1), y.cumsum(dim=1)], dim=-1)


def batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, 4, (batch_size, seq_len), device=device)
    final_xy = coordinates(tokens)[:, -1, :]
    return tokens, final_xy


def x_only_pairs(n: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs with exactly +2 x displacement and identical y.

    The donor replaces a W in the base sequence with E. We force the
    first token to W, then edit that token. The task is order-invariant,
    which means other replacements with the same net effect are valid too.
    """
    base, _ = batch(n, seq_len, device)
    base[:, 0] = 3  # W
    donor = base.clone()
    donor[:, 0] = 1  # E
    return base, donor
