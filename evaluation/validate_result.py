"""Strict validation of an evaluation result file against the paper protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SHIFTS = ("1.0", "2.0", "3.0", "4.0")
EXPECTED_ALARM = "centralized_sum_top_N_two_sided_cusum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--target-arl0", type=float, default=370.0)
    parser.add_argument("--arl0-tolerance", type=float, default=20.0)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=2000)
    return parser.parse_args()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / max(len(values), 1)


def validate_record(
    record: dict[str, Any],
    *,
    target_arl0: float = 370.0,
    arl0_tolerance: float = 20.0,
    replications: int = 1000,
    horizon: int = 2000,
) -> list[str]:
    issues: list[str] = []
    protocol = record.get("protocol", {})
    setting = record.get("setting", {})
    environment = record.get("environment", {})

    if record.get("method") != "dispatchppo":
        issues.append("method must be dispatchppo")
    if int(record.get("schema_version", 0)) < 3:
        issues.append("schema_version must be at least 3")
    if protocol.get("alarm") != EXPECTED_ALARM:
        issues.append(f"alarm must be {EXPECTED_ALARM}")
    if int(protocol.get("replications", -1)) != replications:
        issues.append(f"replications must be {replications}")
    if int(protocol.get("horizon", -1)) != horizon:
        issues.append(f"horizon must be {horizon}")
    if float(protocol.get("target_arl0", math.nan)) != float(target_arl0):
        issues.append(f"target_arl0 must be {target_arl0:g}")
    if protocol.get("change_point") != [5, 50]:
        issues.append("change_point must be [5, 50]")
    if protocol.get("change_point_distribution") != "discrete_uniform":
        issues.append("change point distribution must be discrete_uniform")
    if protocol.get("affected_set") != "uniform_random_subset_per_replication":
        issues.append("affected set must be a uniform random subset per replication")
    if protocol.get("false_alarm_handling") != "skip_prechange_false_alarms":
        issues.append("pre-change false alarms must be skipped")
    if protocol.get("miss_handling") != "remaining_horizon_delay":
        issues.append("misses must receive remaining-horizon delay")
    if protocol.get("threshold_calibration") != "method_specific_null_monte_carlo_bisection":
        issues.append("threshold calibration metadata is incompatible")
    if int(record.get("schema_version", 0)) >= 4:
        if protocol.get("post_change_batching") != "single_gpu_tensor_all_shifts":
            issues.append("schema-4 post-change tests must batch all shifts in one tensor")
        if protocol.get("post_change_common_random_numbers") is not True:
            issues.append("schema-4 shift groups must use common random numbers")
        if protocol.get("shift_batch_order") != [1.0, 2.0, 3.0, 4.0]:
            issues.append("schema-4 shift batch order must be [1, 2, 3, 4]")
        seed_offsets = protocol.get("test_seed_offsets", {})
        if set(seed_offsets) != set(EXPECTED_SHIFTS):
            issues.append("schema-4 test seed offsets must cover all four shifts")
        elif len({int(value) for value in seed_offsets.values()}) != 1:
            issues.append("schema-4 batched shifts must share one test seed offset")

    grid = int(setting.get("grid_size", 0))
    num_uavs = int(setting.get("num_uavs", 0))
    num_anomalies = int(setting.get("num_anomalies", 0))
    if grid not in {5, 10, 30}:
        issues.append("grid_size must be one of 5, 10, 30")
    if not 1 <= num_uavs <= max(grid * grid, 0):
        issues.append("num_uavs is outside the grid capacity")
    if not 1 <= num_anomalies <= max(grid * grid, 0):
        issues.append("num_anomalies is outside the grid capacity")
    expected_uncertainty = 0.001 if grid >= 30 else 0.01
    if not math.isclose(
        float(environment.get("uncertainty_increment", math.nan)),
        expected_uncertainty,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        issues.append(f"uncertainty_increment must be {expected_uncertainty:g}")
    if float(environment.get("mu_min", math.nan)) != 1.0:
        issues.append("mu_min must be 1")
    if environment.get("initial_positions") != "fixed_evenly_spaced_diagonal":
        issues.append("initial positions are incompatible")
    if environment.get("motion") != "collision_free_8_neighbor_plus_stay_no_swaps":
        issues.append("motion protocol is incompatible")
    if environment.get("boundary_moves") != "excluded":
        issues.append("boundary-violating moves must be excluded")

    fingerprint_payload = {
        "protocol": protocol,
        "environment": environment,
        "setting": setting,
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if record.get("protocol_fingerprint") != expected_fingerprint:
        issues.append("protocol_fingerprint does not match the artifact metadata")

    calibration = record.get("calibration", {})
    ic_lengths = calibration.get("running_lengths", [])
    if len(ic_lengths) != replications:
        issues.append(f"calibration must contain {replications} running lengths")
    elif not math.isclose(
        _mean(ic_lengths),
        float(calibration.get("arl0", math.nan)),
        abs_tol=1.0e-9,
    ):
        issues.append("calibration ARL0 does not match its running lengths")
    if not _finite(calibration.get("threshold")) or float(calibration.get("threshold", -1)) < 0:
        issues.append("calibrated threshold must be finite and nonnegative")
    if not _finite(calibration.get("arl0")):
        issues.append("calibrated ARL0 is non-finite")
    elif abs(float(calibration["arl0"]) - target_arl0) > arl0_tolerance:
        issues.append(
            f"realized ARL0={float(calibration['arl0']):.3f} differs from "
            f"target by more than {arl0_tolerance:g}"
        )

    tests = record.get("tests", {})
    if set(tests) != set(EXPECTED_SHIFTS):
        issues.append(f"test shifts must be exactly {list(EXPECTED_SHIFTS)}")
    for shift in EXPECTED_SHIFTS:
        if shift not in tests:
            continue
        metrics = tests[shift]
        alarms = metrics.get("alarm_times", [])
        change_points = metrics.get("change_points", [])
        delays = metrics.get("running_lengths", [])
        valid = int(metrics.get("valid_replications", -1))
        if len(alarms) != replications:
            issues.append(f"delta={shift}: alarm_times must have length {replications}")
        if len(change_points) != replications:
            issues.append(f"delta={shift}: change_points must have length {replications}")
        elif any(int(nu) < 5 or int(nu) > 50 for nu in change_points):
            issues.append(f"delta={shift}: change point outside [5, 50]")
        if len(delays) != valid:
            issues.append(f"delta={shift}: valid_replications does not match stored delays")
        if valid < 1 or valid > replications:
            issues.append(f"delta={shift}: invalid valid_replications={valid}")
        if delays and not math.isclose(
            _mean(delays),
            float(metrics.get("arl1_skip_false", math.nan)),
            abs_tol=1.0e-9,
        ):
            issues.append(f"delta={shift}: ARL1 does not match stored delays")
        for key in (
            "arl1_skip_false",
            "arl1_skip_false_std",
            "arl1_skip_false_se",
            "false_alarm_rate",
            "miss_rate",
            "detection_rate",
            "pre_change_coverage",
            "post_change_hit_rate",
        ):
            if not _finite(metrics.get(key)):
                issues.append(f"delta={shift}: {key} is non-finite")
        for key in (
            "false_alarm_rate",
            "miss_rate",
            "detection_rate",
            "pre_change_coverage",
            "post_change_hit_rate",
        ):
            value = float(metrics.get(key, math.nan))
            if _finite(value) and not 0.0 <= value <= 1.0:
                issues.append(f"delta={shift}: {key} is outside [0, 1]")
        if len(alarms) == replications and len(change_points) == replications:
            false_count = sum(
                int(alarm) <= horizon and int(alarm) < int(nu)
                for alarm, nu in zip(alarms, change_points)
            )
            expected_valid = replications - false_count
            if valid != expected_valid:
                issues.append(
                    f"delta={shift}: false-alarm skip count implies "
                    f"{expected_valid} valid replications, found {valid}"
                )
            false_rate = false_count / replications
            if not math.isclose(
                false_rate,
                float(metrics.get("false_alarm_rate", math.nan)),
                abs_tol=1.0e-7,
            ):
                issues.append(f"delta={shift}: false_alarm_rate is inconsistent")
    return issues


def main() -> None:
    args = parse_args()
    issue_count = 0
    for path in args.paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        issues = validate_record(
            record,
            target_arl0=args.target_arl0,
            arl0_tolerance=args.arl0_tolerance,
            replications=args.replications,
            horizon=args.horizon,
        )
        if issues:
            issue_count += len(issues)
            print(f"[INVALID] {path}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[VALID] {path}")
    if issue_count:
        raise SystemExit(f"{issue_count} validation issue(s)")


if __name__ == "__main__":
    main()
