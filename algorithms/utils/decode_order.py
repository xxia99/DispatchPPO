"""Utilities for constructing per-environment UAV decoding orders."""

from __future__ import annotations

import torch


DECODE_ORDERS = ("fixed", "random", "sorted")


def build_decode_permutation(
    uav_positions: torch.Tensor,
    w_grid: torch.Tensor | None,
    decode_order: str,
) -> torch.Tensor:
    """Return original UAV indices in the order they should be decoded.

    Args:
        uav_positions: Integer coordinates with shape ``[B, N, 2]``.
        w_grid: Per-cell evidence with shape ``[B, G, G]``. Required for
            evidence-sorted decoding.
        decode_order: One of ``fixed``, ``random``, or ``sorted``.
    """

    if uav_positions.ndim != 3 or uav_positions.shape[-1] != 2:
        raise ValueError("uav_positions must have shape [B, N, 2]")
    if decode_order not in DECODE_ORDERS:
        raise ValueError(
            f"decode_order must be one of {DECODE_ORDERS}, got {decode_order!r}"
        )

    batch_size, num_uavs, _ = uav_positions.shape
    device = uav_positions.device
    fixed = torch.arange(num_uavs, dtype=torch.long, device=device)
    if decode_order == "fixed":
        return fixed.view(1, num_uavs).expand(batch_size, num_uavs)
    if decode_order == "random":
        return torch.rand(
            (batch_size, num_uavs),
            device=device,
        ).argsort(dim=1)

    if w_grid is None:
        raise ValueError("sorted decoding requires a CUSUM evidence map")
    if w_grid.ndim != 3 or w_grid.shape[0] != batch_size:
        raise ValueError("w_grid must have shape [B, G, G]")
    batch = torch.arange(batch_size, device=device).unsqueeze(1)
    rows = uav_positions[..., 0].long()
    cols = uav_positions[..., 1].long()
    evidence = w_grid[batch, rows, cols]
    # Stable sorting makes the original UAV index the deterministic tie-break.
    return torch.argsort(evidence, dim=1, descending=True, stable=True)
