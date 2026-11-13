"""Evaluate every saved checkpoint of a training run and tabulate the results.

For each ``candidate_*.pt``, ``step_*.pt`` and ``best_*.pt`` file in the run
directory, ``evaluation.evaluate`` is run (seed 42, 1,000 replications, shifts
1,2,3,4 by default) unless its result JSON already exists; ``--parallel N``
runs N evaluations as concurrent subprocesses. The summary table
``SUMMARY.txt`` lists, per checkpoint, the calibrated threshold H, the realized
ARL0 and the conditional detection delay for each shift.

Run from the repository root::

    python -m evaluation.sweep_checkpoints results/g10n5_run --grid-size 10 \
        --num-uavs 5 --num-anomalies 5 --label-free --decode-order sorted \
        --epsilon-ic 0.01 --parallel 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from algorithms.utils.decode_order import DECODE_ORDERS


CHECKPOINT_PATTERN = re.compile(r"^(candidate_|step_|best_[0-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="directory holding the run's *.pt checkpoints")
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--num-uavs", type=int, required=True)
    parser.add_argument("--num-anomalies", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where the per-checkpoint JSON files and SUMMARY.txt go (default: <run_dir>/sweep42)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--target-arl0", type=float, default=370.0)
    parser.add_argument("--shifts", type=str, default="1,2,3,4")
    parser.add_argument("--change-low", type=int, default=5)
    parser.add_argument("--change-high", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uncertainty-increment", type=float, default=None)
    parser.add_argument("--epsilon-ic", type=float, default=0.01)
    parser.add_argument("--decode-order", choices=DECODE_ORDERS, default="fixed")
    parser.add_argument("--label-free", action="store_true")
    parser.add_argument("--parallel", type=int, default=1, help="number of concurrent evaluation subprocesses")
    return parser.parse_args()


def list_checkpoints(run_dir: Path) -> list[Path]:
    files = sorted(set(glob.glob(str(run_dir / "*.pt"))))
    return [Path(f) for f in files if CHECKPOINT_PATTERN.match(Path(f).name)]


def evaluate_command(args: argparse.Namespace, checkpoint: Path, output: Path, cache_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "evaluation.evaluate",
        "--checkpoint", str(checkpoint),
        "--grid-size", str(args.grid_size),
        "--num-uavs", str(args.num_uavs),
        "--num-anomalies", str(args.num_anomalies),
        "--replications", str(args.replications),
        "--chunk-size", str(args.chunk_size),
        "--horizon", str(args.horizon),
        "--seed", str(args.seed),
        "--target-arl0", str(args.target_arl0),
        "--shifts", args.shifts,
        "--device", args.device,
        "--change-low", str(args.change_low),
        "--change-high", str(args.change_high),
        "--decode-order", args.decode_order,
        "--epsilon-ic", str(args.epsilon_ic),
        "--cache-dir", str(cache_dir),
        "--output", str(output),
    ]
    if args.uncertainty_increment is not None:
        cmd += ["--uncertainty-increment", str(args.uncertainty_increment)]
    if args.label_free:
        cmd.append("--label-free")
    return cmd


def run_one(args: argparse.Namespace, checkpoint: Path, out_dir: Path) -> tuple[str, int]:
    name = checkpoint.stem
    output = out_dir / f"{name}.json"
    if output.exists():
        return name, 0
    cache_dir = out_dir / f"cache_{name}"
    log_path = out_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.call(
            evaluate_command(args, checkpoint, output, cache_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
    shutil.rmtree(cache_dir, ignore_errors=True)
    return name, code


def _step_index(name: str) -> int:
    parts = name.split("_")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1


def write_summary(out_dir: Path, shifts: list[str]) -> str:
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        tests = record["tests"]
        calibration = record["calibration"]
        name = path.stem
        delays = [tests[key]["conditional_delay"] for key in sorted(tests, key=float)]
        rows.append((_step_index(name), name, calibration.get("threshold", 0.0), calibration["arl0"], delays))
    rows.sort()
    header = "ckpt            H      ARL0   " + " ".join(f"d{s:<5s}" for s in shifts)
    lines = [header]
    for _, name, threshold, arl0, delays in rows:
        lines.append("%-15s %6.2f %6.1f " % (name, threshold, arl0) + " ".join("%6.2f" % value for value in delays))
    if rows:
        best = min(rows, key=lambda row: row[4][0])
        lines.append("")
        lines.append(f"best at delta={shifts[0]}: {best[1]} {best[4][0]:.2f}")
    text = "\n".join(lines) + "\n"
    (out_dir / "SUMMARY.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir is not None else args.run_dir / f"sweep{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list_checkpoints(args.run_dir)
    (out_dir / "ckpts.txt").write_text("".join(f"{p}\n" for p in checkpoints), encoding="utf-8")
    print(f"{len(checkpoints)} checkpoints", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as pool:
        for name, code in pool.map(lambda ck: run_one(args, ck, out_dir), checkpoints):
            status = "ok" if code == 0 else f"FAILED (exit {code}, see {name}.log)"
            print(f"[sweep] {name}: {status}", flush=True)
            if code != 0:
                failures.append(name)
    shifts = [s.strip().rstrip("0").rstrip(".") if "." in s else s.strip() for s in args.shifts.split(",") if s.strip()]
    print(write_summary(out_dir, shifts), end="")
    if failures:
        raise SystemExit(f"{len(failures)} evaluation(s) failed: {failures}")


if __name__ == "__main__":
    main()
