from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class BenchRow:
    mode: str
    eval_layering: int
    exit_code: int
    score: float
    total_wall_time_sec: float
    makespan_mean: float
    makespan_p90: float
    feasible_rate: float


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_first_run_makespan(csv_path: Path) -> float:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                return float(row.get("final_ms", "nan"))
            except Exception:
                return float("nan")
    return float("nan")


def _run_once(
    *,
    project_dir: Path,
    repo_root: Path,
    prefix: str,
    gammas: str,
    iters: int,
    seeds: str,
    speed_profile: str,
    time_budget_sec: float,
    no_seed: bool,
    ignore_cache: bool,
    eval_layering: int,
    max_exact_evals_per_iter: int,
    baseline: str,
    tol: float,
) -> BenchRow:
    main_py = project_dir / "main.py"
    summary_path = project_dir / f"bench_quick_{prefix}_summary.json"
    runs_path = project_dir / f"bench_quick_{prefix}_runs.csv"

    cmd: List[str] = [
        sys.executable,
        str(main_py),
        "--quick-bench",
        "--prefix",
        str(prefix),
        "--bench-iters",
        str(int(iters)),
        "--bench-seeds",
        str(seeds),
        "--gammas",
        str(gammas),
        "--alns-speed-profile",
        str(speed_profile),
        "--alns-time-budget-sec",
        str(float(time_budget_sec)),
        "--eval-layering",
        str(int(eval_layering)),
        "--max-exact-evals-per-iter",
        str(int(max_exact_evals_per_iter)),
        "--bench-compare",
        str(baseline),
        "--bench-tol",
        str(float(tol)),
    ]
    if no_seed:
        cmd.append("--alns-no-seed")
    if ignore_cache:
        cmd.append("--alns-ignore-cache")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        cmd,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
    )
    print(proc.stdout)

    summary = _read_json(summary_path)
    by_gamma = summary.get("by_gamma", [])
    gamma0 = by_gamma[0] if by_gamma else {}
    row = BenchRow(
        mode="layering_on" if int(eval_layering) == 1 else "baseline",
        eval_layering=int(eval_layering),
        exit_code=int(proc.returncode),
        score=float(summary.get("overall", {}).get("score", float("nan"))),
        total_wall_time_sec=float(summary.get("total_wall_time_sec", float("nan"))),
        makespan_mean=float(gamma0.get("makespan_mean", float("nan"))),
        makespan_p90=float(gamma0.get("makespan_p90", float("nan"))),
        feasible_rate=float(gamma0.get("feasible_rate", float("nan"))),
    )

    mode_tag = "layering1" if int(eval_layering) == 1 else "layering0"
    shutil.copyfile(summary_path, project_dir / f"bench_quick_{prefix}_summary_{mode_tag}.json")
    shutil.copyfile(runs_path, project_dir / f"bench_quick_{prefix}_runs_{mode_tag}.csv")
    print(
        f"[run_benchmarks] mode={mode_tag} "
        f"score={row.score:.6f} wall={row.total_wall_time_sec:.2f}s "
        f"ms={_read_first_run_makespan(runs_path):.2f} exit={row.exit_code}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="demo01")
    ap.add_argument("--gammas", default="20")
    ap.add_argument("--bench-iters", type=int, default=500)
    ap.add_argument("--bench-seeds", default="0")
    ap.add_argument("--alns-speed-profile", default="balanced", choices=["balanced", "turbo"])
    ap.add_argument("--alns-time-budget-sec", type=float, default=0.0)
    ap.add_argument("--alns-no-seed", action="store_true")
    ap.add_argument("--alns-ignore-cache", action="store_true")
    ap.add_argument("--max-exact-evals-per-iter", type=int, default=1)
    ap.add_argument("--bench-compare", default="bench_baseline_demo01.json")
    ap.add_argument("--bench-tol", type=float, default=0.001)
    ap.add_argument("--out-csv", default="bench_layering_compare.csv")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    project_dir = script_path.parent.parent
    repo_root = project_dir.parent

    rows: List[BenchRow] = []
    for layering in (0, 1):
        rows.append(
            _run_once(
                project_dir=project_dir,
                repo_root=repo_root,
                prefix=str(args.prefix),
                gammas=str(args.gammas),
                iters=int(args.bench_iters),
                seeds=str(args.bench_seeds),
                speed_profile=str(args.alns_speed_profile),
                time_budget_sec=float(args.alns_time_budget_sec),
                no_seed=bool(args.alns_no_seed),
                ignore_cache=bool(args.alns_ignore_cache),
                eval_layering=int(layering),
                max_exact_evals_per_iter=int(args.max_exact_evals_per_iter),
                baseline=str(args.bench_compare),
                tol=float(args.bench_tol),
            )
        )

    out_csv = project_dir / str(args.out_csv)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "eval_layering",
                "exit_code",
                "score",
                "total_wall_time_sec",
                "makespan_mean",
                "makespan_p90",
                "feasible_rate",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.mode,
                    r.eval_layering,
                    r.exit_code,
                    f"{r.score:.8f}",
                    f"{r.total_wall_time_sec:.8f}",
                    f"{r.makespan_mean:.8f}",
                    f"{r.makespan_p90:.8f}",
                    f"{r.feasible_rate:.8f}",
                ]
            )
    print(f"[run_benchmarks] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

