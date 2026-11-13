"""GPU-vectorized evaluation of a DispatchPPO checkpoint under the paper protocol.

Protocol: random affected set per replication, change point
tau ~ Uniform{change_low, ..., change_high}, two-sided CUSUM with uncertainty
inflation, centralized sum-of-top-N alarm, threshold calibrated once to the
target ARL0 on in-control replications (bisection) and then frozen, detection
delay measured on out-of-control replications for every shift (all shifts in
one batch with common random numbers). Replications that alarm before the
change point are skipped; misses receive the remaining-horizon delay.

Run from the repository root::

    python -m evaluation.evaluate --grid-size 10 --num-uavs 5 --num-anomalies 5 \
        --checkpoint /path/to/actor.pt --output results/eval/dispatchppo_10_5_5.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch

from algorithms.utils.decode_order import DECODE_ORDERS
from evaluation.dispatchppo_adapter import DispatchPPOAdapter
from evaluation.gpu_env import EnvConfig, GpuUAVCentralEnv


METHOD = "dispatchppo"
CACHE_SCHEMA_VERSION = 9
RESULT_SCHEMA_VERSION = 4


# Stay bias of the executed policy: xi = sigmoid(STAY_BIAS_SLOPE * (gap - STAY_BIAS_OFFSET))
# with gap = w_cur - max_{k'} w_{k'} the local evidence gap.
STAY_BIAS_SLOPE = 10.0
STAY_BIAS_OFFSET = 0.5

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--num-uavs", type=int, required=True)
    parser.add_argument("--num-anomalies", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="DispatchNet actor checkpoint (.pt)")
    parser.add_argument(
        "--label-free",
        action="store_true",
        help=(
            "Location-ordered UAV numbering in the environment (UAV k occupies "
            "the k-th smallest cell id). Required for checkpoints trained with "
            "--label-free; refused otherwise."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=1000, help="replications per shift in one GPU batch")
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--target-arl0", type=float, default=370.0)
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Skip the ARL0 calibration and evaluate at this threshold; "
                             "the in-control rollout is still run so that the realized ARL0 is reported.")
    parser.add_argument("--shifts", type=str, default="1,2,3,4")
    parser.add_argument("--change-low", type=int, default=5)
    parser.add_argument("--change-high", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uncertainty-increment",
        type=float,
        default=None,
        help="C^U; default 0.001 for grid_size >= 30, else 0.01",
    )
    parser.add_argument("--epsilon-ic", type=float, default=0.01, help="exploration rate of the executed policy")
    parser.add_argument(
        "--decode-order",
        choices=DECODE_ORDERS,
        default="fixed",
        help="Autoregressive UAV ordering used by the decoder",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/eval_cache"))
    parser.add_argument("--reuse-ic-cache", action="store_true")
    return parser.parse_args(argv)


def _uncertainty_increment(args: argparse.Namespace) -> float:
    if args.uncertainty_increment is not None:
        return float(args.uncertainty_increment)
    return 0.001 if int(args.grid_size) >= 30 else 0.01


def validate_batching_args(args: argparse.Namespace) -> None:
    if int(args.replications) < 1:
        raise ValueError("--replications must be positive")
    if int(args.chunk_size) < 1:
        raise ValueError("--chunk-size must be positive")


def _cache_path(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint)
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "size": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "mtime_ns": checkpoint_path.stat().st_mtime_ns if checkpoint_path.exists() else None,
    }
    cache_identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "method": METHOD,
        "checkpoint": checkpoint,
        "device": str(torch.device(args.device)),
        "grid_size": args.grid_size,
        "num_uavs": args.num_uavs,
        "num_anomalies": args.num_anomalies,
        "horizon": args.horizon,
        "replications": args.replications,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "uncertainty_increment": _uncertainty_increment(args),
        "label_free": bool(args.label_free),
        "epsilon_ic": args.epsilon_ic,
        "stay_bias": "sigmoid(%g*(gap-%g))" % (STAY_BIAS_SLOPE, STAY_BIAS_OFFSET),
        "decode_order": args.decode_order,
    }
    digest = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return args.cache_dir / (
        f"ic_{METHOD}_g{args.grid_size}_n{args.num_uavs}_"
        f"h{args.horizon}_r{args.replications}_s{args.seed}_{digest}.pt"
    )


def _check_label_free_consistency(args: argparse.Namespace) -> None:
    """Refuse to evaluate a label-free checkpoint without label-free numbering."""

    payload = torch.load(args.checkpoint, map_location="cpu")
    trained_label_free = False
    if isinstance(payload, dict):
        trained_label_free = bool(
            payload.get("training", {})
            .get("policy_update", {})
            .get("label_free", False)
        )
    if trained_label_free != bool(args.label_free):
        raise ValueError(
            "checkpoint label_free="
            f"{trained_label_free} but --label-free={bool(args.label_free)}; "
            "the policy input must use the numbering it was trained with"
        )


def _chunks(total: int, size: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < total:
        count = min(size, total - start)
        yield start, count
        start += count


def _parse_shifts(text: str) -> list[float]:
    shifts = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not shifts:
        raise ValueError("--shifts must contain at least one value")
    if len(set(shifts)) != len(shifts):
        raise ValueError("--shifts must not contain duplicate values")
    return shifts


def _env_config(args: argparse.Namespace, num_envs: int, shift: float, seed: int) -> EnvConfig:
    return EnvConfig(
        grid_size=args.grid_size,
        num_uavs=args.num_uavs,
        num_anomalies=args.num_anomalies,
        num_envs=num_envs,
        horizon=args.horizon,
        shift=float(shift),
        uncertainty_increment=_uncertainty_increment(args),
        change_low=args.change_low,
        change_high=args.change_high,
        random_anomalies=True,
        fixed_start=True,
        seed=seed,
        label_free=bool(args.label_free),
    )


@torch.no_grad()
def rollout(
    args: argparse.Namespace,
    device: torch.device,
    shift: float,
    seed_offset: int,
    policy: DispatchPPOAdapter,
) -> dict[str, torch.Tensor | float]:
    """In-control (or single-shift) rollout; the environment draws its own measurements."""

    stats_parts = []
    nu_parts = []
    hit_parts = []
    coverage_parts = []
    elapsed = 0.0

    for start, count in _chunks(int(args.replications), int(args.chunk_size)):
        policy_seed = int(args.seed) + int(seed_offset) + start + 500000
        torch.manual_seed(policy_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(policy_seed)
        env = GpuUAVCentralEnv(
            _env_config(args, count, shift, int(args.seed) + int(seed_offset) + start),
            device,
        )
        policy.reset(count)

        stats = torch.empty(
            (count, args.horizon),
            dtype=torch.float32,
            device=device,
        )
        hit_sum = torch.zeros(count, dtype=torch.float32, device=device)
        hit_steps = torch.zeros(count, dtype=torch.float32, device=device)
        pre_seen = torch.zeros((count, env.K), dtype=torch.bool, device=device)
        start_time = time.perf_counter()
        for step in range(args.horizon):
            actions = policy.act(env)
            info = env.step(
                actions,
                resolution_order=policy.last_decode_permutation,
            )
            stats[:, step] = info["global_stat"]
            post = info["post_change"]
            hit_sum += torch.where(post, info["hit_rate"], 0.0)
            hit_steps += post.float()
            pre_seen |= env.z_prev & (~post).unsqueeze(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed += time.perf_counter() - start_time
        stats_parts.append(stats.detach().cpu())
        nu_parts.append(env.nu.detach().cpu())
        hit_parts.append((hit_sum / hit_steps.clamp_min(1.0)).detach().cpu())
        coverage_parts.append(pre_seen.float().mean(dim=1).detach().cpu())

    return {
        "stats": torch.cat(stats_parts, dim=0),
        "nu": torch.cat(nu_parts, dim=0),
        "post_hit_rate": torch.cat(hit_parts, dim=0),
        "pre_coverage": torch.cat(coverage_parts, dim=0),
        "elapsed_seconds": elapsed,
    }


@torch.no_grad()
def rollout_shifts_batched(
    args: argparse.Namespace,
    device: torch.device,
    shifts: list[float],
    seed_offset: int,
    policy: DispatchPPOAdapter,
) -> tuple[dict[float, dict[str, torch.Tensor | float]], float]:
    """Evaluate all post-change shifts in one flattened GPU environment batch.

    For each replication chunk, the leading environment dimension is
    ``[num_shifts, replications_per_shift]`` flattened to one tensor. The
    delta groups share change points, affected sets, and Gaussian innovations
    (common random numbers), while their policies evolve independently after
    the evidence fields diverge.
    """

    if not shifts:
        raise ValueError("shifts must not be empty")
    num_shifts = len(shifts)
    stats_parts: list[list[torch.Tensor]] = [[] for _ in shifts]
    nu_parts: list[list[torch.Tensor]] = [[] for _ in shifts]
    hit_parts: list[list[torch.Tensor]] = [[] for _ in shifts]
    coverage_parts: list[list[torch.Tensor]] = [[] for _ in shifts]
    elapsed = 0.0

    for start, count in _chunks(int(args.replications), int(args.chunk_size)):
        total = num_shifts * count
        policy_seed = int(args.seed) + int(seed_offset) + start + 500000
        torch.manual_seed(policy_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(policy_seed)
        env = GpuUAVCentralEnv(
            _env_config(args, total, 0.0, int(args.seed) + int(seed_offset) + start),
            device,
        )

        # Pair the exogenous scenario across deltas. This improves precision of
        # cross-delta comparisons without coupling the policies themselves.
        env.nu = env.nu[:count].repeat(num_shifts)
        env.anomaly_mask = env.anomaly_mask[:count].repeat(num_shifts, 1)
        env.pos = env.pos[:count].repeat(num_shifts, 1)
        shift_values = torch.tensor(
            shifts,
            dtype=torch.float32,
            device=device,
        ).repeat_interleave(count)

        policy.reset(total)

        stats = torch.empty(
            (num_shifts, count, args.horizon),
            dtype=torch.float32,
            device=device,
        )
        hit_sum = torch.zeros(total, dtype=torch.float32, device=device)
        hit_steps = torch.zeros(total, dtype=torch.float32, device=device)
        pre_seen = torch.zeros((total, env.K), dtype=torch.bool, device=device)
        start_time = time.perf_counter()
        for step in range(args.horizon):
            actions = policy.act(env)

            base_y = torch.randn(
                (count, env.K),
                generator=env.generator,
                dtype=torch.float32,
                device=device,
            )
            measurements = base_y.repeat(num_shifts, 1)
            post = (env.t + 1) >= env.nu
            measurements.add_(
                post.float().unsqueeze(1)
                * env.anomaly_mask.float()
                * shift_values.unsqueeze(1)
            )
            info = env.step(
                actions,
                measurements=measurements,
                resolution_order=policy.last_decode_permutation,
            )
            stats[:, :, step] = info["global_stat"].view(num_shifts, count)
            post = info["post_change"]
            hit_sum += torch.where(post, info["hit_rate"], 0.0)
            hit_steps += post.float()
            pre_seen |= env.z_prev & (~post).unsqueeze(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed += time.perf_counter() - start_time

        nu = env.nu.view(num_shifts, count).detach().cpu()
        hit = (hit_sum / hit_steps.clamp_min(1.0)).view(num_shifts, count).detach().cpu()
        coverage = (
            pre_seen.float()
            .mean(dim=1)
            .view(num_shifts, count)
            .detach()
            .cpu()
        )
        for index in range(num_shifts):
            stats_parts[index].append(stats[index].detach().cpu())
            nu_parts[index].append(nu[index])
            hit_parts[index].append(hit[index])
            coverage_parts[index].append(coverage[index])

    elapsed_share = elapsed / num_shifts
    trajectories = {}
    for index, shift in enumerate(shifts):
        trajectories[shift] = {
            "stats": torch.cat(stats_parts[index], dim=0),
            "nu": torch.cat(nu_parts[index], dim=0),
            "post_hit_rate": torch.cat(hit_parts[index], dim=0),
            "pre_coverage": torch.cat(coverage_parts[index], dim=0),
            "elapsed_seconds": elapsed_share,
        }
    return trajectories, elapsed


def build_policy(args: argparse.Namespace, device: torch.device) -> DispatchPPOAdapter:
    _check_label_free_consistency(args)
    return DispatchPPOAdapter(
        args.checkpoint,
        args.grid_size,
        args.num_uavs,
        device,
        epsilon_ic=args.epsilon_ic,
        beta_mix=STAY_BIAS_SLOPE,
        delta0=STAY_BIAS_OFFSET,
        decode_order=args.decode_order,
    )


def running_lengths(stats: torch.Tensor, threshold: float) -> torch.Tensor:
    crossed = stats >= float(threshold)
    has_crossed = crossed.any(dim=1)
    first = crossed.float().argmax(dim=1).long() + 1
    return torch.where(has_crossed, first, torch.full_like(first, stats.shape[1]))


def calibrate_threshold(
    stats: torch.Tensor,
    target_arl0: float,
) -> tuple[float, torch.Tensor]:
    lo = 0.0
    hi = float(stats.max().item()) + 1.0
    candidates: list[tuple[float, float, torch.Tensor]] = []
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        rl = running_lengths(stats, mid)
        error = abs(float(rl.double().mean().item()) - float(target_arl0))
        candidates.append((error, mid, rl))
        if float(rl.double().mean().item()) < target_arl0:
            lo = mid
        else:
            hi = mid
    _, threshold, rl = min(candidates, key=lambda item: item[0])
    return threshold, rl


def summarize_ic(
    rl: torch.Tensor,
    threshold: float,
    horizon: int,
) -> dict[str, float | list[int]]:
    values = rl.double()
    return {
        "threshold": float(threshold),
        "arl0": float(values.mean().item()),
        "std": float(values.std(unbiased=True).item()),
        "se": float(values.std(unbiased=True).item() / math.sqrt(len(values))),
        "false_alarm_probability_by_horizon": float((rl < int(horizon)).float().mean().item()),
        "running_lengths": rl.tolist(),
    }


def summarize_oc(
    trajectory: dict[str, torch.Tensor | float],
    threshold: float,
) -> dict[str, float | int | list[int]]:
    stats = trajectory["stats"]
    nu = trajectory["nu"].long()
    assert isinstance(stats, torch.Tensor)
    assert isinstance(nu, torch.Tensor)
    crossed = stats >= float(threshold)
    has_crossed = crossed.any(dim=1)
    alarm = crossed.float().argmax(dim=1).long() + 1
    alarm = torch.where(has_crossed, alarm, torch.full_like(alarm, stats.shape[1] + 1))
    false_alarm = has_crossed & (alarm < nu)
    no_alarm = ~has_crossed
    valid = ~false_alarm
    delay = torch.where(
        no_alarm,
        stats.shape[1] - nu + 1,
        alarm - nu + 1,
    )
    valid_delay = delay[valid].double()
    detected_after = has_crossed & (alarm >= nu)
    conditional = delay[detected_after].double()
    valid_std = (
        float(valid_delay.std(unbiased=True).item())
        if len(valid_delay) > 1
        else 0.0
    )
    return {
        "arl1_skip_false": float(valid_delay.mean().item()),
        "arl1_skip_false_std": valid_std,
        "arl1_skip_false_se": float(valid_std / math.sqrt(max(len(valid_delay), 1))),
        "conditional_delay": float(conditional.mean().item()) if len(conditional) else math.nan,
        "conditional_delay_std": float(conditional.std(unbiased=True).item()) if len(conditional) > 1 else math.nan,
        "false_alarm_rate": float(false_alarm.float().mean().item()),
        "miss_rate": float(no_alarm.float().mean().item()),
        "valid_replications": int(valid.sum().item()),
        "detection_rate": float(detected_after.float().mean().item()),
        "pre_change_coverage": float(trajectory["pre_coverage"].double().mean().item()),
        "post_change_hit_rate": float(trajectory["post_hit_rate"].double().mean().item()),
        "running_lengths": delay[valid].tolist(),
        "alarm_times": alarm.tolist(),
        "change_points": nu.tolist(),
    }


def _checkpoint_metadata(path_text: str | None) -> dict | None:
    if not path_text:
        return None
    path = Path(path_text)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    validate_batching_args(args)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy = build_policy(args, device)

    cache_path = _cache_path(args)
    if args.reuse_ic_cache and cache_path.exists():
        ic_trajectory = torch.load(cache_path, map_location="cpu")
        print(f"[eval] reused IC trajectory {cache_path}", flush=True)
    else:
        ic_trajectory = rollout(
            args,
            device,
            shift=0.0,
            seed_offset=10000,
            policy=policy,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ic_trajectory, cache_path)
        print(f"[eval] saved IC trajectory {cache_path}", flush=True)
    if args.fixed_threshold is not None:
        threshold = float(args.fixed_threshold)
        ic_rl = running_lengths(ic_trajectory["stats"], threshold)
    else:
        threshold, ic_rl = calibrate_threshold(ic_trajectory["stats"], args.target_arl0)
    calibration = summarize_ic(ic_rl, threshold, args.horizon)
    calibration["elapsed_seconds"] = float(ic_trajectory["elapsed_seconds"])
    calibration["seconds_per_batched_step"] = float(
        ic_trajectory["elapsed_seconds"] / max(args.horizon, 1)
    )
    print(
        f"[eval] {METHOD} G={args.grid_size} N={args.num_uavs}: "
        f"H={threshold:.6f}, ARL0={calibration['arl0']:.3f}",
        flush=True,
    )

    shifts = _parse_shifts(args.shifts)
    tests = {}
    trajectories, joint_oc_elapsed = rollout_shifts_batched(
        args,
        device,
        shifts=shifts,
        seed_offset=20000,
        policy=policy,
    )
    for shift in shifts:
        trajectory = trajectories[shift]
        metrics = summarize_oc(trajectory, threshold)
        metrics["elapsed_seconds"] = float(trajectory["elapsed_seconds"])
        metrics["seconds_per_batched_step"] = float(
            trajectory["elapsed_seconds"] / max(args.horizon, 1)
        )
        tests[str(shift)] = metrics
        print(
            f"[eval] delta={shift:g}: ARL1={metrics['arl1_skip_false']:.3f} "
            f"SE={metrics['arl1_skip_false_se']:.3f} "
            f"FA={metrics['false_alarm_rate']:.3f} hit={metrics['post_change_hit_rate']:.3f}",
            flush=True,
        )

    protocol = {
        "alarm": "centralized_sum_top_N_two_sided_cusum",
        "target_arl0": args.target_arl0,
        "replications": args.replications,
        "replication_batch_size": args.chunk_size,
        "replication_batching": "accelerator_tensor_chunks",
        "cpu_multiprocessing": False,
        "accelerator_resident_rollout_statistics": True,
        "horizon": args.horizon,
        "change_point": [args.change_low, args.change_high],
        "change_point_distribution": "discrete_uniform",
        "affected_set": "uniform_random_subset_per_replication",
        "false_alarm_handling": "skip_prechange_false_alarms",
        "miss_handling": "remaining_horizon_delay",
        "threshold_calibration": "method_specific_null_monte_carlo_bisection",
        "calibration_only": False,
        "calibration_seed_offset": 10000,
        "post_change_batching": "single_gpu_tensor_all_shifts",
        "post_change_common_random_numbers": True,
        "shift_batch_order": shifts,
        "test_seed_offsets": {
            str(float(value)): 20000
            for value in shifts
        },
    }
    environment = {
        "uncertainty_increment": _uncertainty_increment(args),
        "mu_min": 1.0,
        "initial_positions": "fixed_evenly_spaced_diagonal",
        "motion": "collision_free_8_neighbor_plus_stay_no_swaps",
        "boundary_moves": "excluded",
    }
    if bool(args.label_free):
        # Only label-free runs carry the key, so the fingerprints of the
        # physical-index records are unchanged.
        environment["uav_numbering"] = "location_ordered"
    fingerprint_payload = {
        "protocol": protocol,
        "environment": environment,
        "setting": {
            "grid_size": args.grid_size,
            "num_uavs": args.num_uavs,
            "num_anomalies": args.num_anomalies,
        },
    }
    protocol_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "protocol": protocol,
        "method": METHOD,
        "setting": {
            "grid_size": args.grid_size,
            "num_uavs": args.num_uavs,
            "num_anomalies": args.num_anomalies,
        },
        "checkpoint": _checkpoint_metadata(args.checkpoint),
        "environment": environment,
        "policy": {
            "epsilon_ic": args.epsilon_ic,
            "stay_bias_slope": STAY_BIAS_SLOPE,
            "stay_bias_offset": STAY_BIAS_OFFSET,
            "decode_order": args.decode_order,
        },
        "timing": {
            "post_change_joint_elapsed_seconds": float(joint_oc_elapsed),
        },
        "calibration": calibration,
        "tests": tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[eval] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
