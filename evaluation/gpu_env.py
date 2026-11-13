"""GPU-vectorized environment for the paper's evaluation protocol.

This module implements the protocol stated in the paper:

* two-sided Gaussian tabular CUSUM;
* uncertainty inflation of unvisited locations;
* collision-free 8-neighbor-plus-stay motion; and
* a centralized sum-of-top-N alarm statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


MOVES = torch.tensor(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ],
    dtype=torch.long,
)


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int
    num_uavs: int
    num_anomalies: int
    num_envs: int
    horizon: int = 2000
    shift: float = 0.0
    mu_min: float = 1.0
    uncertainty_increment: float = 0.01
    change_low: int = 5
    change_high: int = 50
    random_anomalies: bool = True
    # Optional per-environment random affected-set size |B| ~ Uniform{low..high}
    # (inclusive), redrawn at every reset; overrides num_anomalies when set.
    num_anomalies_low: int | None = None
    num_anomalies_high: int | None = None
    fixed_change_point: int | None = None
    fixed_start: bool = True
    validate_runtime_invariants: bool = True
    seed: int = 1
    # Location-ordered UAV numbering: after every reset and step the UAV
    # indices are re-assigned so that index k occupies the k-th smallest cell
    # id.  The policy input is then a function of the observed set alone (a
    # label-free policy).  The environment has no per-UAV state other than
    # ``pos``, so sorting is the whole change.
    label_free: bool = False


class GpuUAVCentralEnv:
    """Batched UAV monitoring state and transitions."""

    def __init__(self, config: EnvConfig, device: torch.device):
        self.config = config
        self.device = device
        self.G = int(config.grid_size)
        self.K = self.G * self.G
        self.N = int(config.num_uavs)
        self.A_count = int(config.num_anomalies)
        if config.num_anomalies_low is not None:
            lo = int(config.num_anomalies_low)
            hi = int(config.num_anomalies_high if config.num_anomalies_high is not None else lo)
            if not 1 <= lo <= hi <= self.K:
                raise ValueError(f"num_anomalies range must satisfy 1 <= low <= high <= {self.K}, got {lo}..{hi}")
        self.E = int(config.num_envs)
        self.horizon = int(config.horizon)
        if not 1 <= self.N <= self.K:
            raise ValueError(f"num_uavs must be in [1, {self.K}], got {self.N}")
        if not 0 <= self.A_count <= self.K:
            raise ValueError(
                f"num_anomalies must be in [0, {self.K}], "
                f"got {self.A_count}"
            )
        if config.change_low > config.change_high:
            raise ValueError("change_low must not exceed change_high")
        if config.change_high > config.horizon:
            raise ValueError("change_high must not exceed the evaluation horizon")

        self.moves = MOVES.to(device)
        self.node_r = torch.arange(self.K, device=device) // self.G
        self.node_c = torch.arange(self.K, device=device) % self.G
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(config.seed))
        self.reset()

    def reset(self) -> torch.Tensor:
        cfg = self.config
        self.t = 0
        # Per-environment clock; equals ``t`` unless environments are reset
        # individually with ``reset_envs`` (training with alarm-terminated
        # episodes).
        self.t_env = torch.zeros(self.E, dtype=torch.long, device=self.device)
        self.W_pos = torch.zeros((self.E, self.K), dtype=torch.float32, device=self.device)
        self.W_neg = torch.zeros_like(self.W_pos)
        self.S = torch.zeros_like(self.W_pos)
        self.age = torch.zeros_like(self.W_pos)
        self.z_prev = torch.zeros_like(self.W_pos, dtype=torch.bool)
        self.last_y = torch.zeros_like(self.W_pos)

        if cfg.fixed_change_point is None:
            self.nu = torch.randint(
                int(cfg.change_low),
                int(cfg.change_high) + 1,
                (self.E,),
                generator=self.generator,
                device=self.device,
            )
        else:
            self.nu = torch.full(
                (self.E,),
                int(cfg.fixed_change_point),
                dtype=torch.long,
                device=self.device,
            )
        self.anomaly_mask = self._sample_anomaly_masks()
        self.pos = self._initial_positions()
        self._apply_label_free_order()
        self._assert_unique_positions()
        self.pos_acting = self.pos.clone()
        return self.build_dispatch_obs()

    def _apply_label_free_order(self) -> None:
        """Re-number the UAVs by the cell id they occupy (label-free mode)."""
        if self.config.label_free:
            self.pos = torch.sort(self.pos, dim=1).values

    def reset_envs(self, mask: torch.Tensor) -> None:
        """Restart the environments selected by ``mask`` with a fresh scenario.

        Used for alarm-terminated training episodes: an environment whose
        system statistic crossed the alarm threshold restarts from the initial
        configuration with zero statistics, zero ages, a new change point, and
        a new affected set, while the other environments continue.
        """

        mask = mask.to(device=self.device, dtype=torch.bool)
        if not bool(mask.any()):
            return
        rows = mask.nonzero(as_tuple=False).squeeze(1)
        count = int(rows.numel())
        zero = torch.zeros((count, self.K), dtype=torch.float32, device=self.device)
        self.W_pos[rows] = zero
        self.W_neg[rows] = zero
        self.S[rows] = zero
        self.age[rows] = zero
        self.z_prev[rows] = False
        self.last_y[rows] = zero
        self.t_env[rows] = 0
        cfg = self.config
        if cfg.fixed_change_point is None:
            self.nu[rows] = torch.randint(
                int(cfg.change_low), int(cfg.change_high) + 1, (count,),
                generator=self.generator, device=self.device,
            )
        else:
            self.nu[rows] = int(cfg.fixed_change_point)
        full_mask = self._sample_anomaly_masks()
        self.anomaly_mask[rows] = full_mask[rows]
        self.pos[rows] = self._initial_positions()[rows]
        self._apply_label_free_order()
        self._assert_unique_positions()

    def _initial_positions(self) -> torch.Tensor:
        if self.config.fixed_start:
            # Evenly spaced diagonal initialization (valid for N <= G).
            if self.N <= self.G:
                idx = torch.linspace(0, self.G - 1, self.N, device=self.device).long()
                flat = idx * self.G + idx
            else:
                # Fleets larger than the grid side: evenly spaced distinct
                # cells in row-major order (N <= K), the same deterministic
                # start in every environment.
                flat = torch.linspace(0, self.K - 1, self.N, device=self.device).round().long()
            return flat.view(1, self.N).expand(self.E, self.N).clone()
        scores = torch.rand(
            (self.E, self.K),
            generator=self.generator,
            device=self.device,
        )
        return torch.topk(scores, k=self.N, dim=1).indices

    def _sample_anomaly_masks(self) -> torch.Tensor:
        mask = torch.zeros((self.E, self.K), dtype=torch.bool, device=self.device)
        if self.A_count == 0:
            return mask
        if self.config.random_anomalies:
            scores = torch.rand(
                (self.E, self.K),
                generator=self.generator,
                device=self.device,
            )
            if self.config.num_anomalies_low is not None:
                low = int(self.config.num_anomalies_low)
                high = int(self.config.num_anomalies_high if self.config.num_anomalies_high is not None else low)
                counts = torch.randint(
                    low, high + 1, (self.E,), generator=self.generator, device=self.device
                )
                ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)
                return ranks < counts.view(-1, 1)
            idx = torch.topk(scores, k=self.A_count, dim=1).indices
        else:
            idx = torch.arange(self.A_count, device=self.device).view(1, -1)
            idx = idx.expand(self.E, self.A_count)
        mask.scatter_(1, idx, True)
        return mask

    def candidate_cells(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidate cell IDs and in-bound flags, both shaped [E,N,9]."""
        move = self.moves.view(1, 1, 9, 2)
        r0 = self.node_r[self.pos].view(self.E, self.N, 1)
        c0 = self.node_c[self.pos].view(self.E, self.N, 1)
        raw_r = r0 + move[..., 0]
        raw_c = c0 + move[..., 1]
        in_bounds = (raw_r >= 0) & (raw_r < self.G) & (raw_c >= 0) & (raw_c < self.G)
        r = raw_r.clamp(0, self.G - 1)
        c = raw_c.clamp(0, self.G - 1)
        return r * self.G + c, in_bounds

    def get_action_masks(self) -> torch.Tensor:
        """Feasible local actions before resolving simultaneous proposals.

        A UAV may remain in its own cell but cannot enter a cell occupied by a
        different UAV. This matches the autoregressive decoder's no-swap
        feasibility convention.
        """
        targets, in_bounds = self.candidate_cells()
        occupancy = torch.zeros((self.E, self.K), dtype=torch.bool, device=self.device)
        occupancy.scatter_(1, self.pos, True)
        occupied = torch.gather(
            occupancy.unsqueeze(1).expand(self.E, self.N, self.K),
            2,
            targets,
        )
        own = targets == self.pos.unsqueeze(-1)
        masks = in_bounds & (~occupied | own)
        masks[..., 4] = True
        return masks

    def resolve_actions(
        self,
        actions: torch.Tensor,
        resolution_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute proposals with the requested autoregressive collision order.

        Invalid proposals fall back to stay. Earlier UAVs cannot take the
        current cells of later-decoded UAVs, so the stay fallback remains free.
        """
        actions = actions.squeeze(-1).long().clamp(0, 8)
        targets, in_bounds = self.candidate_cells()
        chosen = torch.empty((self.E, self.N), dtype=torch.long, device=self.device)
        executed = torch.empty_like(chosen)
        assigned = torch.zeros((self.E, self.K), dtype=torch.bool, device=self.device)
        batch = torch.arange(self.E, device=self.device)
        if resolution_order is None:
            resolution_order = torch.arange(
                self.N,
                dtype=torch.long,
                device=self.device,
            ).view(1, self.N).expand(self.E, self.N)
        else:
            resolution_order = resolution_order.to(
                device=self.device,
                dtype=torch.long,
            )
            if resolution_order.shape != (self.E, self.N):
                raise ValueError(
                    "resolution_order must have shape "
                    f"{(self.E, self.N)}, got {tuple(resolution_order.shape)}"
                )
            expected = torch.arange(
                self.N,
                dtype=torch.long,
                device=self.device,
            ).view(1, self.N)
            if (
                self.config.validate_runtime_invariants
                and not torch.equal(
                    torch.sort(resolution_order, dim=1).values,
                    expected.expand(self.E, self.N),
                )
            ):
                raise ValueError(
                    "each resolution_order row must be a permutation of UAV indices"
                )

        for k in range(self.N):
            agent_idx = resolution_order[:, k]
            cands = targets[batch, agent_idx]
            invalid = ~in_bounds[batch, agent_idx]
            invalid |= torch.gather(assigned, 1, cands)
            if k < self.N - 1:
                future_idx = resolution_order[:, k + 1 :]
                future = torch.gather(self.pos, 1, future_idx)
                invalid |= (cands.unsqueeze(2) == future.unsqueeze(1)).any(dim=2)

            proposed_action = actions[batch, agent_idx]
            proposed_invalid = invalid[batch, proposed_action]
            action_i = torch.where(
                proposed_invalid,
                torch.full_like(proposed_action, 4),
                proposed_action,
            )
            cell_i = cands[batch, action_i]
            if (
                self.config.validate_runtime_invariants
                and bool(
                    torch.gather(
                        assigned,
                        1,
                        cell_i.unsqueeze(1),
                    ).any().item()
                )
            ):
                raise RuntimeError("collision resolver produced a duplicate cell")
            chosen[batch, agent_idx] = cell_i
            executed[batch, agent_idx] = action_i
            assigned[batch, cell_i] = True
        return executed.unsqueeze(-1), chosen

    def step(
        self,
        actions: torch.Tensor,
        measurements: Optional[torch.Tensor] = None,
        resolution_order: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        executed, self.pos = self.resolve_actions(
            actions,
            resolution_order=resolution_order,
        )
        # ``executed`` is indexed by the pre-step UAV numbering; the numbering
        # is refreshed only after the move has been resolved.  ``pos_acting``
        # keeps the post-move cells in that pre-step order (the order in which
        # the policy's slots acted), for per-slot reward credit.
        self.pos_acting = self.pos.clone()
        self._apply_label_free_order()
        self.t += 1
        self.t_env += 1

        z = torch.zeros((self.E, self.K), dtype=torch.bool, device=self.device)
        z.scatter_(1, self.pos, True)
        post = self.t_env >= self.nu
        if measurements is None:
            y = torch.randn(
                (self.E, self.K),
                generator=self.generator,
                dtype=torch.float32,
                device=self.device,
            )
            y = y + (
                post.float().unsqueeze(1)
                * self.anomaly_mask.float()
                * float(self.config.shift)
            )
        else:
            if measurements.shape != (self.E, self.K):
                raise ValueError(
                    f"measurements must have shape {(self.E, self.K)}, "
                    f"got {tuple(measurements.shape)}"
                )
            y = measurements.to(device=self.device, dtype=torch.float32)

        ref = 0.5 * float(self.config.mu_min)
        scaled_y = float(self.config.mu_min) * y
        inflation = float(self.config.uncertainty_increment)
        self.W_pos = torch.where(
            z,
            torch.relu(self.W_pos + scaled_y - ref * float(self.config.mu_min)),
            self.W_pos + inflation,
        )
        self.W_neg = torch.where(
            z,
            torch.relu(self.W_neg - scaled_y - ref * float(self.config.mu_min)),
            self.W_neg + inflation,
        )
        self.S = torch.maximum(self.W_pos, self.W_neg)
        self.age = torch.where(z, torch.zeros_like(self.age), self.age + 1.0)
        self.z_prev = z
        self.last_y = y

        top = torch.topk(self.S, k=self.N, dim=1)
        global_stat = top.values.sum(dim=1)
        hit = (z & self.anomaly_mask).sum(dim=1).float()
        hit = hit / self.anomaly_mask.sum(dim=1).clamp_min(1).float()
        return {
            "actions": executed,
            "positions": self.pos,
            "sampled": z,
            "post_change": post,
            "global_stat": global_stat,
            "hit_rate": hit,
        }

    def build_dispatch_obs(self) -> torch.Tensor:
        """Return the three DispatchNet channels ``[CUSUM, uncertainty, occupancy]``."""
        occupancy = torch.zeros((self.E, self.K), dtype=torch.float32, device=self.device)
        occupancy.scatter_(1, self.pos, 1.0)
        uncertainty = self.age * float(self.config.uncertainty_increment)
        channels = [
            self.S.view(self.E, self.G, self.G),
            uncertainty.view(self.E, self.G, self.G),
            occupancy.view(self.E, self.G, self.G),
        ]
        return torch.stack(channels, dim=1)

    def position_xy(self) -> torch.Tensor:
        return torch.stack([self.node_r[self.pos], self.node_c[self.pos]], dim=2)

    def _assert_unique_positions(self) -> None:
        ordered = torch.sort(self.pos, dim=1).values
        duplicate = (ordered[:, 1:] == ordered[:, :-1]).any()
        if bool(duplicate.item()):
            raise RuntimeError("UAV positions are not unique")
