"""Accelerator-vectorized PPO training of the DispatchNet policy (DispatchPPO).

Every training environment and policy rollout lives in one accelerator tensor
(``evaluation.gpu_env.GpuUAVCentralEnv``).  Measurements follow the paper's
in-control model (iid N(0, 1) per location) with the prespecified mean shifts
of ``--training-shifts`` cycled across the environment batch after a random
change point.  The policy is updated with clipped PPO on the executed policy's
own joint-move probabilities; a separate convolutional critic estimates the
state value.  Checkpoint selection runs the paper's evaluation protocol
(``evaluation.evaluate``: threshold calibration to the target ARL0, then ARL1
at shift 1) in a subprocess every ``--validate-every`` updates.

Typical use::

    python -m training.make_scratch_checkpoint --grid-size 10 --num-uavs 5 \
        --output runs/g10n5/scratch_init.pt --seed 101
    python -m training.train_dispatch_gpu --grid-size 10 --num-uavs 5 \
        --num-anomalies 5 --init-checkpoint runs/g10n5/scratch_init.pt \
        --output-dir runs/g10n5 ...   (see scripts/train_gpu.sh)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from algorithms.utils.decode_order import build_decode_permutation
from evaluation.dispatchppo_adapter import (
    DispatchPPOAdapter,
    _unwrap_state_dict,
    infer_actor_metadata,
    masked_dispatch_actions,
)
from evaluation.gpu_env import EnvConfig, GpuUAVCentralEnv
from evaluation.evaluate import STAY_BIAS_SLOPE, STAY_BIAS_OFFSET

# Training rollouts (and the PPO ratio) use the executed policy of Section 4.3: the learned
# distribution with the stay bias and the epsilon_IC mixture, with the same constants as
# the evaluation.  Validation and test use epsilon_IC = 0.01 as well.
EXECUTION_EPSILON_IC = 0.01



# Largest fleet for which the observed-set KL diagnostic enumerates all N!
# relabelings; larger fleets log the joint-move KL only.
RELABEL_MAX_UAVS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--num-uavs", type=int, required=True)
    parser.add_argument("--num-anomalies", type=int, required=True, help="Training affected-set size |B|.")
    parser.add_argument("--init-checkpoint", type=Path, required=True,
                        help="DispatchNet actor checkpoint to start from (make_scratch_checkpoint.py or an earlier run's latest.pt).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-actor-optimizer",
        action="store_true",
        help="Restore the actor Adam state embedded in --init-checkpoint (exact continuation), keeping the requested --learning-rate.",
    )
    parser.add_argument(
        "--resume-critic",
        action="store_true",
        help="Restore the critic weights and critic Adam state embedded in --init-checkpoint (exact continuation).",
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=None,
                        help="Cap this process's CUDA allocator at this fraction of the device memory (torch.cuda.set_per_process_memory_fraction).")
    parser.add_argument("--num-envs", type=int, default=224, help="Parallel training environments (one accelerator tensor).")
    parser.add_argument("--episode-steps", type=int, default=96, help="Training episode length.")
    parser.add_argument("--ppo-updates", type=int, default=80)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--num-mini-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--uncertainty-increment", type=float, default=0.01, help="C^U, the per-step inflation of unvisited locations.")
    parser.add_argument("--change-low", type=int, default=5, help="Change point tau ~ Uniform{change-low, ..., change-high}.")
    parser.add_argument("--change-high", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save candidate_XXXX.pt every K updates.")
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--training-shifts",
        default="1,2,3,4",
        help="Comma-separated post-change mean shifts assigned cyclically across the training environments.",
    )
    parser.add_argument(
        "--decode-order",
        choices=("fixed", "sorted"),
        default="fixed",
        help="Stage-II autoregressive collision-resolution order. ``sorted`` decodes UAVs on higher-CUSUM cells first.",
    )
    parser.add_argument(
        "--label-free",
        action="store_true",
        help="Location-ordered UAV numbering: after every step UAV k occupies the k-th smallest cell id (policy input depends on the observed set only).",
    )
    parser.add_argument(
        "--divergence-log-samples",
        type=int,
        default=512,
        help="Rollout states used after each PPO update to estimate the joint-move and observed-set KL between the old and the new policy (0 disables).",
    )
    # --- in-training validation, checkpoint naming, logging, lr schedule ---
    parser.add_argument(
        "--validate-every",
        type=int,
        default=0,
        help=(
            "Validate every K PPO updates with evaluation.evaluate (calibrate to the target ARL0, test at delta=1 on the "
            "validation seed). The checkpoint is saved as best_XXXX.pt when its validation ARL1 beats every earlier one "
            "in the run, else step_XXXX.pt. 0 disables."
        ),
    )
    parser.add_argument("--validate-replications", type=int, default=1000)
    parser.add_argument("--validate-chunk-size", type=int, default=250)
    parser.add_argument("--validate-seed", type=int, default=271828)
    parser.add_argument("--validate-horizon", type=int, default=2000)
    parser.add_argument("--validate-target-arl0", type=float, default=370.0)
    parser.add_argument("--validate-num-anomalies", type=int, default=None,
                        help="Affected-set size for validation (the test protocol's |B|); default = the training --num-anomalies.")
    parser.add_argument("--validate-fixed-threshold", type=float, default=None,
                        help="Validate at this fixed threshold instead of calibrating to --validate-target-arl0.")
    parser.add_argument("--value-loss-form", choices=("mse", "huber"), default="mse",
                        help="Critic loss: MSE or Huber with --huber-delta.")
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--validate-change-low", type=int, default=5)
    parser.add_argument("--validate-change-high", type=int, default=50)
    parser.add_argument("--swanlab-project", type=str, default=None, help="SwanLab project; unset disables logging.")
    parser.add_argument("--swanlab-name", type=str, default=None)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="disabled")
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Linear decay of the actor learning rate to this value at the last update.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Stop the PPO epochs of an update as soon as the minibatch estimate of KL(pi_old || pi_new) (k3) exceeds this value.",
    )
    parser.add_argument(
        "--log-ratio-clamp",
        type=float,
        default=10.0,
        help="Symmetric clamp on the log probability ratio before exponentiation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "grid_size",
        "num_uavs",
        "num_anomalies",
        "num_envs",
        "episode_steps",
        "num_mini_batches",
        "checkpoint_interval",
    )
    for name in positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.num_uavs <= args.grid_size**2:
        raise ValueError("--num-uavs exceeds the grid capacity")
    if not 1 <= args.num_anomalies <= args.grid_size**2:
        raise ValueError("--num-anomalies exceeds the grid capacity")
    if not 1 <= args.change_low <= args.change_high <= args.episode_steps:
        raise ValueError("change range must lie inside the training episode")
    if args.ppo_updates < 1:
        raise ValueError("--ppo-updates must be positive")
    training_shifts = [
        float(value.strip())
        for value in getattr(args, "training_shifts", "1,2,3,4").split(",")
        if value.strip()
    ]
    if not training_shifts or any(value <= 0.0 for value in training_shifts):
        raise ValueError("--training-shifts must contain positive values")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DispatchValueNet(nn.Module):
    """Small centralized critic used only by the PPO optimizer."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features(obs)
        pooled = torch.cat(
            [
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.value(pooled).squeeze(1)


def build_obs(env: GpuUAVCentralEnv) -> torch.Tensor:
    return env.build_dispatch_obs()


def actor_logits(
    actor: nn.Module,
    obs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    features = actor.base(obs)
    encoded = actor.encoder(features, positions)
    return actor.act.decoder(encoded)


def dispatch_target_feasibility(
    env: GpuUAVCentralEnv,
    actions: torch.Tensor,
    *,
    decode_order: str = "fixed",
) -> torch.Tensor:
    """Identify joint actions representable by the autoregressive decoder."""

    actions = actions.squeeze(-1).long()
    if actions.shape != (env.E, env.N):
        raise ValueError(
            f"actions must have shape {(env.E, env.N)}, "
            f"got {tuple(actions.shape)}"
        )
    if bool(((actions < 0) | (actions >= 9)).any().item()):
        raise ValueError("actions must use the nine DispatchNet move indices")

    candidates, in_bounds = env.candidate_cells()
    assigned = torch.zeros(
        (env.E, env.K),
        dtype=torch.bool,
        device=env.device,
    )
    feasible = torch.ones(env.E, dtype=torch.bool, device=env.device)
    batch = torch.arange(env.E, device=env.device)
    decode_permutation = build_decode_permutation(
        env.position_xy(),
        env.S.view(env.E, env.G, env.G),
        decode_order,
    )
    for decode_index in range(env.N):
        agent = decode_permutation[:, decode_index]
        agent_candidates = candidates[batch, agent]
        agent_in_bounds = in_bounds[batch, agent]
        invalid = ~agent_in_bounds
        invalid |= torch.gather(assigned, 1, agent_candidates)
        if decode_index < env.N - 1:
            future_agent = decode_permutation[:, decode_index + 1 :]
            future = torch.gather(env.pos, 1, future_agent)
            invalid |= (
                agent_candidates.unsqueeze(2) == future.unsqueeze(1)
            ).any(dim=2)

        action = actions[batch, agent]
        feasible &= ~invalid[batch, action]
        cell = agent_candidates[batch, action]
        assigned[batch, cell] = True
    return feasible


def _permutations(num: int, device: torch.device) -> torch.Tensor:
    """All permutations of ``range(num)`` as a ``[num!, num]`` tensor."""

    import itertools

    perms = list(itertools.permutations(range(num)))
    return torch.tensor(perms, dtype=torch.long, device=device)


def _clone_mask_env(
    env: GpuUAVCentralEnv,
    rows: torch.Tensor,
) -> GpuUAVCentralEnv:
    """A mask-only environment holding the selected rows of ``env``."""

    config = EnvConfig(**{**asdict(env.config), "num_envs": int(rows.numel())})
    clone = GpuUAVCentralEnv(config, env.device)
    clone.pos = env.pos[rows]
    clone.S = env.S[rows]
    return clone


def observed_set_log_prob(
    env: GpuUAVCentralEnv,
    logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    epsilon_ic: float = 0.0,
    beta_mix: float = 1.0,
    delta0: float = 0.0,
    decode_order: str = "fixed",
    max_uavs: int = RELABEL_MAX_UAVS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of the observed set of a joint move (diagnostic).

    The observed set of a joint move is the set of cells the fleet occupies
    next.  Its probability under the policy is the sum of the joint-move
    probabilities over every feasible relabeling: joint moves that send the
    UAVs to the same cells in a different assignment.  Relabelings are
    enumerated over all ``N!`` permutations, so the fleet size is limited by
    ``max_uavs``.

    Returns ``(log_q, num_feasible)`` with ``log_q`` of shape ``[E]`` and
    ``num_feasible`` the number of feasible relabelings per row.
    """

    if env.N > int(max_uavs):
        raise ValueError(
            f"observed-set probability enumerates N! relabelings; N={env.N} "
            f"exceeds max_uavs={max_uavs}"
        )
    actions = actions.squeeze(-1).long()
    if actions.shape != (env.E, env.N):
        raise ValueError(
            f"actions must have shape {(env.E, env.N)}, got {tuple(actions.shape)}"
        )
    candidates, in_bounds = env.candidate_cells()  # [E, N, 9]
    batch = torch.arange(env.E, device=env.device)
    # Cells reached by the sampled move, indexed by UAV.
    cells = candidates[batch.unsqueeze(1), torch.arange(env.N, device=env.device).unsqueeze(0), actions]
    perms = _permutations(env.N, env.device)  # [P, N]
    num_perms = perms.shape[0]
    # Under relabeling sigma, UAV n must reach cells[sigma(n)].
    targets = cells[:, perms]  # [E, P, N]
    reach = candidates.unsqueeze(1) == targets.unsqueeze(-1)  # [E, P, N, 9]
    reach &= in_bounds.unsqueeze(1)
    reachable = reach.any(dim=-1).all(dim=-1)  # [E, P]
    relabeled = reach.float().argmax(dim=-1)  # [E, P, N] move indices

    rows = torch.arange(env.E, device=env.device).repeat_interleave(num_perms)
    flat_actions = relabeled.reshape(env.E * num_perms, env.N)
    flat_reachable = reachable.reshape(env.E * num_perms)
    mask_env = _clone_mask_env(env, rows)
    feasible = flat_reachable & dispatch_target_feasibility(
        mask_env,
        flat_actions.unsqueeze(-1),
        decode_order=decode_order,
    )
    keep = feasible.nonzero(as_tuple=False).squeeze(1)
    score_env = _clone_mask_env(env, rows[keep])
    _, log_probs, _ = masked_dispatch_actions(
        score_env,
        logits[rows[keep]],
        actions=flat_actions[keep].unsqueeze(-1),
        epsilon_ic=epsilon_ic,
        beta_mix=beta_mix,
        delta0=delta0,
        decode_order=decode_order,
    )
    table = torch.full(
        (env.E * num_perms,),
        -torch.inf,
        dtype=log_probs.dtype,
        device=env.device,
    )
    table = table.index_put((keep,), log_probs.sum(dim=1))
    log_q = torch.logsumexp(table.view(env.E, num_perms), dim=1)
    num_feasible = feasible.view(env.E, num_perms).sum(dim=1)
    return log_q, num_feasible


def _gpu_memory_metrics(device: torch.device) -> dict[str, float]:
    """Per-update GPU memory bookkeeping (GB): allocated, reserved, peak allocated since the last call."""
    if device.type != "cuda":
        return {}
    out = {
        "mem/allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
        "mem/reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
        "mem/max_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    torch.cuda.reset_peak_memory_stats(device)
    return out


def _policy_log_probs(
    args: argparse.Namespace,
    actor: nn.Module,
    obs: torch.Tensor,
    positions: torch.Tensor,
    statistics: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Joint-move and (for small fleets) observed-set log-probabilities of actions."""

    env = _mask_env_from_positions(args, positions, device, statistics=statistics)
    logits = actor_logits(actor, obs, positions)
    _, log_probs, _ = masked_dispatch_actions(
        env,
        logits,
        actions=actions,
        epsilon_ic=EXECUTION_EPSILON_IC,
        beta_mix=STAY_BIAS_SLOPE,
        delta0=STAY_BIAS_OFFSET,
        decode_order=args.decode_order,
    )
    joint = log_probs.sum(dim=1)
    if args.num_uavs > RELABEL_MAX_UAVS:
        return joint, None
    log_q, _ = observed_set_log_prob(
        env,
        logits,
        actions,
        epsilon_ic=EXECUTION_EPSILON_IC,
        beta_mix=STAY_BIAS_SLOPE,
        delta0=STAY_BIAS_OFFSET,
        decode_order=args.decode_order,
        max_uavs=RELABEL_MAX_UAVS,
    )
    return joint, log_q


@torch.no_grad()
def divergence_diagnostics(
    args: argparse.Namespace,
    actor: nn.Module,
    old_actor: nn.Module,
    rollout: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """KL estimates between the behavior policy and the updated policy.

    ``kl_joint``: joint-move KL(pi_old || pi_new), k1 estimator
    ``E[log pi_old(a) - log pi_new(a)]`` on the sampled moves, and the
    lower-variance k3 estimator ``E[r - 1 - log r]`` with ``r = pi_new/pi_old``.
    ``kl_observed_set``: the same k1 estimator on the observed-set law
    ``q(V|x)``; ``kl_assignment`` is their difference (the relabeling term).
    """

    actor.eval()
    time_steps, num_envs = rollout["rewards"].shape
    samples = time_steps * num_envs
    count = min(int(getattr(args, "divergence_log_samples", 512)), samples)
    index = torch.linspace(0, samples - 1, count, device=device).long()
    obs = rollout["obs"].reshape(samples, 3, args.grid_size, args.grid_size)[index]
    pos = rollout["positions"].reshape(samples, args.num_uavs, 2)[index]
    stat = rollout["statistics"].reshape(samples, args.grid_size, args.grid_size)[index]
    act = rollout["actions"].reshape(samples, args.num_uavs, 1)[index]
    new_joint, new_log_q = _policy_log_probs(args, actor, obs, pos, stat, act, device)
    old_j, old_log_q = _policy_log_probs(args, old_actor, obs, pos, stat, act, device)
    k1 = old_j - new_joint
    log_ratio = (new_joint - old_j).clamp(-20.0, 20.0)
    k3 = log_ratio.exp() - 1.0 - log_ratio
    out = {
        "divergence/kl_joint_k1": float(k1.mean().item()),
        "divergence/kl_joint_k1_se": float((k1.std(unbiased=True) / math.sqrt(count)).item()),
        "divergence/kl_joint_k3": float(k3.mean().item()),
        "divergence/samples": float(count),
    }
    if old_log_q is not None and new_log_q is not None:
        k1_q = old_log_q - new_log_q
        out["divergence/kl_observed_set_k1"] = float(k1_q.mean().item())
        out["divergence/kl_observed_set_k1_se"] = float(
            (k1_q.std(unbiased=True) / math.sqrt(count)).item()
        )
        out["divergence/kl_assignment_k1"] = float((k1 - k1_q).mean().item())
    actor.train()
    return out


def _make_env(
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
) -> GpuUAVCentralEnv:
    return GpuUAVCentralEnv(
        EnvConfig(
            grid_size=args.grid_size,
            num_uavs=args.num_uavs,
            num_anomalies=args.num_anomalies,
            num_envs=args.num_envs,
            horizon=args.episode_steps,
            shift=0.0,
            uncertainty_increment=args.uncertainty_increment,
            change_low=args.change_low,
            change_high=args.change_high,
            random_anomalies=True,
            fixed_start=True,
            seed=seed,
            label_free=bool(getattr(args, "label_free", False)),
        ),
        device,
    )


def _training_shifts(args: argparse.Namespace, env: GpuUAVCentralEnv) -> torch.Tensor:
    """Per-environment post-change mean shift, cycled from --training-shifts."""

    shifts = torch.tensor(
        [
            float(value.strip())
            for value in getattr(args, "training_shifts", "1,2,3,4").split(",")
            if value.strip()
        ],
        dtype=torch.float32,
        device=env.device,
    )
    shift_index = torch.arange(env.E, device=env.device) % len(shifts)
    return shifts[shift_index]


def _measurements(env: GpuUAVCentralEnv, shifts: torch.Tensor) -> torch.Tensor:
    """iid N(0, 1) measurements at every location, shifted on the affected set after the change point."""

    measurements = torch.randn(
        (env.E, env.K),
        generator=env.generator,
        dtype=torch.float32,
        device=env.device,
    )
    post = (env.t_env + 1) >= env.nu
    return measurements + (
        post.float().unsqueeze(1)
        * env.anomaly_mask.float()
        * shifts.unsqueeze(1)
    )


def shaped_reward_terms(
    env: GpuUAVCentralEnv,
    post_change: torch.Tensor,
    *,
    uncertainty_increment: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two terms of the paper's training reward, per environment.

    Uncertainty penalty = -(C^U / K) sum_k u_k (u_k = age of location k, which
    never exceeds the episode horizon); anomaly term =
    (1/N) sum_n (2 * 1{p_n in B} - 1) after the change point.
    """

    affected = torch.gather(env.anomaly_mask, 1, env.pos).float()
    anomaly_reward = affected.mul(2.0).sub(1.0).mean(dim=1)
    anomaly_reward = anomaly_reward * post_change.float()
    max_age = max(float(env.horizon), 1.0)
    coverage_penalty = -(
        float(uncertainty_increment) * env.age.clamp(max=max_age).mean(dim=1)
    )
    return coverage_penalty, anomaly_reward


@torch.no_grad()
def collect_ppo_rollout(
    args: argparse.Namespace,
    actor: nn.Module,
    critic: DispatchValueNet,
    device: torch.device,
    *,
    update: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    actor.eval()
    critic.eval()
    env = _make_env(args, device, seed=args.seed + 100000 + 1009 * update)
    shifts = _training_shifts(args, env)
    observations = []
    positions = []
    statistics = []
    actions = []
    old_log_probs = []
    values = []
    rewards = []
    post_changes = []
    hit_sum = torch.zeros(env.E, device=device)
    hit_steps = torch.zeros_like(hit_sum)
    pre_seen = torch.zeros((env.E, env.K), dtype=torch.bool, device=device)
    coverage_sum = anomaly_sum = age_sum = 0.0
    started = time.perf_counter()

    for step_index in range(args.episode_steps):
        obs = build_obs(env)
        pos = env.position_xy()
        statistic = env.S.clone()
        logits = actor_logits(actor, obs, pos)
        action, log_probs, _ = masked_dispatch_actions(
            env,
            logits,
            deterministic=False,
            epsilon_ic=EXECUTION_EPSILON_IC,
            beta_mix=STAY_BIAS_SLOPE,
            delta0=STAY_BIAS_OFFSET,
            decode_order=args.decode_order,
        )
        value = critic(obs)
        info = env.step(
            action,
            measurements=_measurements(env, shifts),
        )
        coverage_term, anomaly_term = shaped_reward_terms(
            env,
            info["post_change"],
            uncertainty_increment=args.uncertainty_increment,
        )
        reward = coverage_term + anomaly_term
        coverage_sum += float(coverage_term.mean().item())
        anomaly_sum += float(anomaly_term.mean().item())
        age_sum += float(env.age.mean().item())
        observations.append(obs)
        positions.append(pos)
        statistics.append(statistic)
        actions.append(action)
        old_log_probs.append(log_probs.sum(dim=1))
        values.append(value)
        rewards.append(reward)
        post = info["post_change"]
        post_changes.append(post)
        hit_sum += torch.where(post, info["hit_rate"], 0.0)
        hit_steps += post.float()
        pre_seen |= env.z_prev & (~post).unsqueeze(1)

    elapsed = time.perf_counter() - started
    rollout = {
        "obs": torch.stack(observations),
        "positions": torch.stack(positions),
        "statistics": torch.stack(statistics),
        "actions": torch.stack(actions),
        "old_log_probs": torch.stack(old_log_probs),
        "values": torch.stack(values),
        "rewards": torch.stack(rewards),
        "post_change": torch.stack(post_changes),
    }
    return rollout, {
        "phase": "ppo",
        "train/reward": float(rollout["rewards"].mean().item()),
        "train/reward_coverage": coverage_sum / max(args.episode_steps, 1),
        "train/reward_anomaly": anomaly_sum / max(args.episode_steps, 1),
        "train/mean_age": age_sum / max(args.episode_steps, 1),
        "train/post_hit_rate": float(
            (hit_sum / hit_steps.clamp_min(1.0)).mean().item()
        ),
        "train/pre_coverage": float(pre_seen.float().mean().item()),
        "speed/env_steps_per_second": args.num_envs * args.episode_steps / elapsed,
    }


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    dones: torch.Tensor | None = None,
    last_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE; ``dones[t]`` marks steps after which the environment restarted.

    ``last_value`` bootstraps the truncated tail of episodes that did not end.
    """

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    next_value = (
        torch.zeros_like(values[0]) if last_value is None else last_value
    )
    for step in reversed(range(rewards.shape[0])):
        continuing = 1.0 - (dones[step].float() if dones is not None else 0.0)
        delta = rewards[step] + gamma * next_value * continuing - values[step]
        gae = delta + gamma * gae_lambda * continuing * gae
        advantages[step] = gae
        next_value = values[step]
    return advantages, advantages + values


def _mask_env_from_positions(
    args: argparse.Namespace,
    positions: torch.Tensor,
    device: torch.device,
    statistics: torch.Tensor | None = None,
) -> GpuUAVCentralEnv:
    """Environment holding only the state the collision-feasibility mask and
    the stay bias need (positions and CUSUM statistics)."""

    env = GpuUAVCentralEnv(
        EnvConfig(
            grid_size=args.grid_size,
            num_uavs=args.num_uavs,
            num_anomalies=args.num_anomalies,
            num_envs=len(positions),
            horizon=1,
            change_low=1,
            change_high=1,
            seed=args.seed,
            label_free=bool(getattr(args, "label_free", False)),
        ),
        device,
    )
    env.pos = (
        positions[..., 0].long() * args.grid_size + positions[..., 1].long()
    )
    # Stored positions of a label-free rollout are already location-ordered.
    if statistics is not None:
        expected = (len(positions), args.grid_size, args.grid_size)
        if statistics.shape != expected:
            raise ValueError(
                f"statistics must have shape {expected}, "
                f"got {tuple(statistics.shape)}"
            )
        env.S = statistics.reshape(len(positions), -1).to(
            device=device,
            dtype=torch.float32,
        )
    return env


def ppo_update(
    args: argparse.Namespace,
    actor: nn.Module,
    critic: DispatchValueNet,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    # The actor stays in eval mode: DispatchNet's Transformer layers carry
    # dropout (p = 0.1), and scoring the rollout actions in train mode gives
    # log-probabilities that differ from the rollout's, which corrupts the PPO
    # ratio and trips the KL early stop on the very first minibatch.  Gradients
    # flow in eval mode as usual.
    actor.eval()
    critic.train()
    clamp_value = float(getattr(args, "log_ratio_clamp", 10.0))
    target_kl = getattr(args, "target_kl", None)
    stopped_early = False
    old_actor = copy.deepcopy(actor).eval()
    for parameter in old_actor.parameters():
        parameter.requires_grad_(False)
    advantages, returns = generalized_advantage_estimate(
        rollout["rewards"],
        rollout["values"],
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1.0e-5)

    time_steps, num_envs = rollout["rewards"].shape
    samples = time_steps * num_envs
    flat_obs = rollout["obs"].reshape(samples, 3, args.grid_size, args.grid_size)
    flat_positions = rollout["positions"].reshape(samples, args.num_uavs, 2)
    flat_statistics = rollout["statistics"].reshape(
        samples,
        args.grid_size,
        args.grid_size,
    )
    flat_actions = rollout["actions"].reshape(samples, args.num_uavs, 1)
    flat_old_log_probs = rollout["old_log_probs"].reshape(samples)
    flat_advantages = advantages.reshape(samples)
    flat_returns = returns.reshape(samples)
    flat_actor_active = torch.ones(
        samples,
        dtype=torch.bool,
        device=flat_obs.device,
    )
    mini_batch_size = max(
        math.ceil(samples / max(args.num_mini_batches, 1)),
        1,
    )
    totals = {
        "loss/policy": 0.0,
        "loss/value": 0.0,
        "loss/entropy": 0.0,
        "loss/ratio": 0.0,
        "loss/actor_grad_norm": 0.0,
        "loss/critic_grad_norm": 0.0,
        "loss/clip_fraction": 0.0,
        "loss/max_abs_log_ratio": 0.0,
    }
    actor_count = 0
    critic_count = 0
    clip_range = float(args.clip_param)

    epochs_run = 0
    for _ in range(args.ppo_epochs):
        if stopped_early:
            break
        epochs_run += 1
        permutation = torch.randperm(samples, device=flat_obs.device)
        for start in range(0, samples, mini_batch_size):
            if stopped_early:
                break
            index = permutation[start : start + mini_batch_size]
            obs = flat_obs[index]
            pos = flat_positions[index]
            statistic = flat_statistics[index]
            action = flat_actions[index]
            actor_active = flat_actor_active[index]
            if bool(actor_active.any().item()):
                actor_obs = obs[actor_active]
                actor_pos = pos[actor_active]
                actor_statistic = statistic[actor_active]
                actor_action = action[actor_active]
                # Recreate the state needed by the collision-feasibility mask.
                batch_env = _mask_env_from_positions(
                    args,
                    actor_pos,
                    flat_obs.device,
                    statistics=actor_statistic,
                )
                logits = actor_logits(actor, actor_obs, actor_pos)
                _, log_probs, entropy = masked_dispatch_actions(
                    batch_env,
                    logits,
                    actions=actor_action,
                    epsilon_ic=EXECUTION_EPSILON_IC,
                    beta_mix=STAY_BIAS_SLOPE,
                    delta0=STAY_BIAS_OFFSET,
                    decode_order=args.decode_order,
                )
                joint_log_prob = log_probs.sum(dim=1)
                log_ratio = (
                    joint_log_prob
                    - flat_old_log_probs[index][actor_active]
                ).clamp(-clamp_value, clamp_value)
                ratio = log_ratio.exp()
                advantage = flat_advantages[index][actor_active]
                clipped_ratio = ratio.clamp(
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                surrogate = torch.minimum(
                    ratio * advantage,
                    clipped_ratio * advantage,
                )
                policy_loss = -surrogate.mean()
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - float(args.entropy_coef) * entropy_mean

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(
                    actor.parameters(), args.max_grad_norm
                )
                actor_optimizer.step()
                totals["loss/policy"] += float(policy_loss.item())
                totals["loss/entropy"] += float(entropy_mean.item())
                totals["loss/ratio"] += float(ratio.mean().item())
                totals["loss/actor_grad_norm"] += float(actor_grad.item())
                actor_count += 1
                with torch.no_grad():
                    # Per-state KL estimate (k3) of KL(pi_old || pi_new).
                    k3 = ratio - 1.0 - log_ratio
                    approx_kl = float(k3.mean().item())
                    totals["loss/clip_fraction"] += float(
                        ((ratio - 1.0).abs() > clip_range).float().mean().item()
                    )
                    totals["loss/max_abs_log_ratio"] = max(
                        totals["loss/max_abs_log_ratio"], float(log_ratio.abs().max().item())
                    )
                # The KL is measured on the policy before this minibatch's step
                # and the stop is applied after it.
                if target_kl is not None and approx_kl > float(target_kl):
                    stopped_early = True

            value = critic(obs)
            if str(getattr(args, "value_loss_form", "mse")) == "huber":
                value_loss = F.huber_loss(value, flat_returns[index], delta=float(args.huber_delta))
            else:
                value_loss = 0.5 * (value - flat_returns[index]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            (args.value_loss_coef * value_loss).backward()
            critic_grad = nn.utils.clip_grad_norm_(
                critic.parameters(), args.max_grad_norm
            )
            critic_optimizer.step()

            totals["loss/value"] += float(value_loss.item())
            totals["loss/critic_grad_norm"] += float(critic_grad.item())
            critic_count += 1

    actor_keys = {
        "loss/policy",
        "loss/entropy",
        "loss/ratio",
        "loss/actor_grad_norm",
        "loss/clip_fraction",
    }
    metrics = {
        key: value / max(actor_count if key in actor_keys else critic_count, 1)
        for key, value in totals.items()
        if key != "loss/max_abs_log_ratio"
    }
    metrics["loss/max_abs_log_ratio"] = totals["loss/max_abs_log_ratio"]
    # Critic fit on this rollout before the update: explained variance of the
    # (pre-update) values against the critic targets, and the target spread.
    with torch.no_grad():
        _ret = returns.reshape(-1).float()
        _res = _ret - rollout["values"].reshape(-1).float()
        _var = _ret.var(unbiased=False).clamp_min(1.0e-8)
        metrics["loss/value_explained_variance"] = float(1.0 - _res.var(unbiased=False) / _var)
        metrics["loss/value_target_std"] = float(_var.sqrt())
    metrics["train/ppo_epochs_run"] = float(epochs_run)
    metrics["train/ppo_minibatch_steps"] = float(actor_count)
    metrics["train/kl_early_stop"] = float(stopped_early)
    metrics["train/ppo_actor_active_fraction"] = float(
        flat_actor_active.float().mean().item()
    )
    if int(getattr(args, "divergence_log_samples", 0)) > 0:
        metrics.update(
            divergence_diagnostics(args, actor, old_actor, rollout, device)
        )
    return metrics


def run_validation(
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
) -> dict[str, float]:
    """Calibrate to the target ARL0 and test at delta=1 on the validation seed.

    Runs ``evaluation.evaluate`` in a subprocess on the same device with the
    executed-policy settings used for training, and returns the validation
    ARL1, its SE, the threshold, and the realized ARL0.
    """

    import subprocess
    import sys

    command = [
        sys.executable, "-m", "evaluation.evaluate",
        "--checkpoint", str(checkpoint),
        "--grid-size", str(args.grid_size),
        "--num-uavs", str(args.num_uavs),
        "--num-anomalies", str(getattr(args, "validate_num_anomalies", None) or args.num_anomalies),
        "--replications", str(args.validate_replications),
        "--chunk-size", str(args.validate_chunk_size),
        "--horizon", str(args.validate_horizon),
        "--seed", str(args.validate_seed),
        "--target-arl0", str(args.validate_target_arl0),
        "--shifts", "1",
        "--device", args.device,
        "--uncertainty-increment", str(args.uncertainty_increment),
        # Checkpoint selection always uses the test law Uniform{5,...,50},
        # whatever range training draws from.
        "--change-low", str(getattr(args, "validate_change_low", 5)),
        "--change-high", str(getattr(args, "validate_change_high", 50)),
        "--decode-order", str(args.decode_order),
        "--epsilon-ic", str(EXECUTION_EPSILON_IC),
        "--cache-dir", str(output.parent / "cache"),
        "--output", str(output),
    ]
    if getattr(args, "validate_fixed_threshold", None) is not None:
        command += ["--fixed-threshold", str(args.validate_fixed_threshold)]
    if getattr(args, "label_free", False):
        command.append("--label-free")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    record = json.loads(output.read_text())
    test = record["tests"][[k for k in record["tests"] if float(k) == 1.0][0]]
    return {
        "val/arl1": float(test["arl1_skip_false"]),
        "val/arl1_se": float(test["arl1_skip_false_se"]),
        "val/false_alarm_rate": float(test["false_alarm_rate"]),
        "val/threshold": float(record["calibration"]["threshold"]),
        "val/arl0": float(record["calibration"]["arl0"]),
        "val/post_change_hit_rate": float(test["post_change_hit_rate"]),
    }


def _checkpoint_payload(
    args: argparse.Namespace,
    actor: nn.Module,
    critic: DispatchValueNet,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    update: int,
    phase: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "actor_type": "dispatchnet",
        "state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "training": {
            "update": int(update),
            "phase": phase,
            "setting": {
                "grid_size": args.grid_size,
                "num_uavs": args.num_uavs,
                "num_anomalies": args.num_anomalies,
            },
            "policy_update": {
                "label_free": bool(getattr(args, "label_free", False)),
                "clip_param": float(args.clip_param),
                "target_kl": getattr(args, "target_kl", None),
                "decode_order": str(args.decode_order),
            },
            "noise_conditions": [
                {
                    "name": "iid_standard_gaussian",
                    "distribution": "normal",
                    "mean": 0.0,
                    "standard_deviation": 1.0,
                }
            ],
            "reward": {
                "form": "paper_uncertainty_penalty_plus_postchange_hit",
                "uncertainty_increment": args.uncertainty_increment,
            },
            "execution": {
                "training_rollouts": "executed policy: stay bias sigmoid(%g*(gap-%g)) and epsilon_IC mixture" % (STAY_BIAS_SLOPE, STAY_BIAS_OFFSET),
                "epsilon_ic": EXECUTION_EPSILON_IC,
                "beta_mix": STAY_BIAS_SLOPE,
                "delta0": STAY_BIAS_OFFSET,
            },
            "vectorization": {
                "environment_batch": "single_accelerator_tensor",
                "cpu_multiprocessing": False,
                "num_envs": args.num_envs,
                "device": args.device,
            },
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "metrics": metrics,
        },
    }


def _save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    actor: nn.Module,
    critic: DispatchValueNet,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    update: int,
    phase: str,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            args,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            update=update,
            phase=phase,
            metrics=metrics,
        ),
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cpu":
        print("[DispatchPPO] running on CPU; the paper's runs used CUDA (expect a slow run)", flush=True)
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = torch.load(args.init_checkpoint, map_location="cpu")
    if getattr(args, "cuda_memory_fraction", None) is not None and device.type == "cuda":
        _dev_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(args.cuda_memory_fraction), _dev_index)
        print(f"[memory] per-process CUDA cap {float(args.cuda_memory_fraction):.2f} of {torch.cuda.get_device_properties(_dev_index).total_memory/2**30:.0f} GB", flush=True)
    metadata = infer_actor_metadata(_unwrap_state_dict(checkpoint_payload))
    adapter = DispatchPPOAdapter(
        args.init_checkpoint,
        args.grid_size,
        args.num_uavs,
        device,
        epsilon_ic=0.0,
        stochastic=True,
        decode_order=args.decode_order,
    )
    actor = adapter.actor
    actor.train()
    critic = DispatchValueNet(channels=3).to(device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=args.learning_rate, eps=1.0e-5
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_learning_rate, eps=1.0e-5
    )
    if args.resume_critic:
        if not isinstance(checkpoint_payload, dict) or "critic_state_dict" not in checkpoint_payload:
            raise ValueError("--resume-critic requires a checkpoint payload containing critic_state_dict")
        critic.load_state_dict(checkpoint_payload["critic_state_dict"])
        critic_optimizer.load_state_dict(checkpoint_payload["critic_optimizer_state_dict"])
        for parameter_group in critic_optimizer.param_groups:
            parameter_group["lr"] = args.critic_learning_rate
    if args.resume_actor_optimizer:
        if not isinstance(checkpoint_payload, dict):
            raise ValueError(
                "--resume-actor-optimizer requires a checkpoint payload "
                "containing actor_optimizer_state_dict"
            )
        optimizer_state = checkpoint_payload.get(
            "actor_optimizer_state_dict"
        )
        if not isinstance(optimizer_state, dict):
            raise ValueError(
                "--resume-actor-optimizer was requested, but the checkpoint "
                "does not contain actor_optimizer_state_dict"
            )
        actor_optimizer.load_state_dict(optimizer_state)
        for parameter_group in actor_optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate

    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "actor_metadata": metadata,
        "setting": {
            "grid_size": args.grid_size,
            "num_uavs": args.num_uavs,
            "num_anomalies": args.num_anomalies,
        },
        "training_seed": args.seed,
        "validation_seed": args.validate_seed,
        "vectorization": {
            "environment_batch": "single_accelerator_tensor",
            "cpu_multiprocessing": False,
            "num_envs": args.num_envs,
            "device": str(device),
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_path = args.output_dir / "history.jsonl"
    swanlab_run = None
    if args.swanlab_project and args.swanlab_mode != "disabled":
        import swanlab  # optional dependency, imported only when logging is requested

        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            name=args.swanlab_name or args.output_dir.name,
            mode=args.swanlab_mode,
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            tags=[
                f"G{args.grid_size}N{args.num_uavs}",
                "label_free" if getattr(args, "label_free", False) else "physical_index",
            ],
        )
    _save_checkpoint(
        args.output_dir / "initial.pt",
        args,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        update=0,
        phase="initial",
        metrics={},
    )

    best_hit = -math.inf
    total_updates = args.ppo_updates
    started = time.perf_counter()
    best_val = math.inf
    for update in range(1, total_updates + 1):
        if args.lr_final is not None and total_updates > 1:
            fraction = (update - 1) / (total_updates - 1)
            lr_now = args.learning_rate + fraction * (args.lr_final - args.learning_rate)
            for parameter_group in actor_optimizer.param_groups:
                parameter_group["lr"] = lr_now
        rollout, rollout_metrics = collect_ppo_rollout(
            args,
            actor,
            critic,
            device,
            update=update,
        )
        loss_metrics = ppo_update(
            args,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            rollout,
            device,
        )
        metrics = {**rollout_metrics, **loss_metrics}
        phase = "ppo"

        metrics["time/elapsed_seconds"] = time.perf_counter() - started
        if device.type == "cuda":
            metrics["gpu/max_memory_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
        validate_now = (
            int(args.validate_every) > 0
            and (update % int(args.validate_every) == 0 or update == total_updates)
        )
        if validate_now:
            pending = args.output_dir / f"pending_{update:04d}.pt"
            _save_checkpoint(
                pending, args, actor, critic, actor_optimizer, critic_optimizer,
                update=update, phase=phase, metrics=metrics,
            )
            val_dir = args.output_dir / "validation"
            val_dir.mkdir(parents=True, exist_ok=True)
            val_metrics = run_validation(args, pending, val_dir / f"val_{update:04d}.json")
            improved = val_metrics["val/arl1"] < best_val
            if improved:
                best_val = val_metrics["val/arl1"]
            name = ("best" if improved else "step") + f"_{update:04d}.pt"
            pending.rename(args.output_dir / name)
            val_metrics["val/best_arl1"] = best_val
            val_metrics["val/checkpoint"] = name
            metrics.update(val_metrics)
            print(
                f"[DispatchPPO] validation update={update}: ARL1={val_metrics['val/arl1']:.3f}"
                f"+/-{val_metrics['val/arl1_se']:.3f} H={val_metrics['val/threshold']:.3f} "
                f"ARL0={val_metrics['val/arl0']:.1f} -> {name}",
                flush=True,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"update": update, **metrics, **_gpu_memory_metrics(device)},
                    sort_keys=True,
                )
                + "\n"
            )

        hit_metric = float(metrics.get("train/post_hit_rate", -math.inf))
        if hit_metric > best_hit:
            best_hit = hit_metric
            _save_checkpoint(
                args.output_dir / "best_train_hit.pt",
                args,
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                update=update,
                phase=phase,
                metrics=metrics,
            )
        if (not validate_now) and (update % args.checkpoint_interval == 0 or update == total_updates):
            _save_checkpoint(
                args.output_dir / f"candidate_{update:04d}.pt",
                args,
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                update=update,
                phase=phase,
                metrics=metrics,
            )
        if swanlab_run is not None:
            swanlab_run.log(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)},
                step=update,
            )
        _save_checkpoint(
            args.output_dir / "latest.pt",
            args,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            update=update,
            phase=phase,
            metrics=metrics,
        )
        printable = " ".join(
            f"{key}={value:.5g}"
            for key, value in sorted(metrics.items())
            if isinstance(value, (int, float))
            and key.startswith(("train/", "loss/", "speed/"))
        )
        print(
            f"[DispatchPPO] update={update}/{total_updates} "
            f"phase={phase} {printable}",
            flush=True,
        )

    if swanlab_run is not None:
        swanlab_run.finish()
    print(f"[DispatchPPO] checkpoints: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
