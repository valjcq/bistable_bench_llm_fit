"""Batch runner and summary: conditions x tasks x seeds -> one CSV of results + a summary table.

    python -m bench.batch CONFIG --tasks tasks/toy_*.json --conditions A B:cma B:tpe C:cma C:tpe
        --seeds 0-4 [--out runs/] [--csv results.csv]

A condition is `A`, or `B:<optimizer>` / `C:<optimizer>`; the label in the tables is `B-cma`.

How the summary numbers are defined
- success rate: fraction of runs whose SUBMITTED parameters pass the final check
  (evaluation_metric.success), with a Wilson 95% interval.
- evals to first success: the index (1-based) of the first evaluation, in call order, whose
  parameters pass the checker; the median is over the runs that have one (its n is printed). It
  is computed after the fact from events.jsonl, not by the optimizer, and the checks done for it
  are NOT counted against any budget (they are analysis, not part of the episode). To avoid
  checking every point, only evaluations with score <= tol are candidates (naive and expert_v1;
  for expert_v2, whose score is not on the tol scale, every point is), and each candidate is
  confirmed by the checker (a score is an average over targets, so it is
  never taken as proof of success). A success whose score is > tol is therefore not credited
  (conservative; it cannot happen with the expert objective, whose unmatched targets cost >= 1).
- succeeded by 25% / 50% / 100% of the budget: fraction of runs whose confirmed first success
  is at evaluation <= ceil(fraction * budget).
- mean final error: over all runs, the mean over targets of the submitted parameters' error,
  with an unmatched target counted as 1.0 (the largest possible error for rates in [0, 1]).
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from bench.checker import check
from bench.config import BudgetConfig, ExperimentConfig, load_config
from bench.logging import read_events
from bench.runner import CONDITIONS, run_baseline, run_scripted
from bench.tasks import Task, load_task

RESULT_FIELDS = [
    "label",
    "condition",
    "optimizer",
    "task_id",
    "seed",
    "success",
    "evals_used",
    "first_success_eval",
    "budget",
    "final_error",
    "wall_clock_s",
    "run_name",
]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1, small n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def final_error(metric: dict[str, Any]) -> float:
    """Mean over targets of the final per-target error; an unmatched target counts as 1.0."""
    errs = [1.0 if e is None else e for e in metric["target_errors"]]
    return sum(errs) / len(errs)


def first_success_eval(cfg: ExperimentConfig, task: Task, events_path: Path) -> int | None:
    """Evaluations spent when the first point passing the checker was evaluated (see module docs).

    For a baseline this is the 1-based position of the point; for the scripted agent it also
    counts the agent's own check calls that came before it.
    """
    model = cfg.model()
    tol = cfg.task.tol
    total = cfg.budget.max_evaluations
    evals = [e for e in read_events(events_path) if e["name"] == "evaluate"]
    for e in evals:
        i = (
            total - e["budget_after"]
        )  # evaluations spent once this one is done (agent checks count)
        # expert_v2 scores are not comparable with tol (a success can score above it), so there
        # every evaluated point is a candidate; for the other objectives score <= tol selects.
        if e["arguments"]["objective"] != "expert_v2" and e["output"]["value"] > tol:
            continue  # not a candidate; a candidate must still be confirmed by the checker
        if check(model, e["arguments"]["params"], task.targets, cfg.checker).success:
            return i
    return None


def parse_conditions(specs: Iterable[str]) -> list[tuple[str, str | None]]:
    """['A', 'B:cma', 'S'] -> [('A', None), ('B', 'cma'), ('S', None)]. S = scripted dummy agent."""
    out = []
    for s in specs:
        cond, _, opt = s.partition(":")
        if cond not in (*CONDITIONS, "S") or (cond in ("A", "S")) != (opt == ""):
            raise ValueError(f"bad condition {s!r}: use A, S, B:<optimizer> or C:<optimizer>")
        out.append((cond, opt or None))
    return out


def parse_seeds(text: str) -> list[int]:
    """'0-4' -> [0..4]; '1,5,9' -> [1, 5, 9]."""
    seeds: list[int] = []
    for part in text.split(","):
        lo, dash, hi = part.partition("-")
        seeds += range(int(lo), int(hi) + 1) if dash else [int(lo)]
    return seeds


def _one_run(
    job: tuple[ExperimentConfig, Path, str, str | None, int, Path],
) -> dict[str, Any]:
    """Run one (task, condition, seed) and return its result row. Top-level so it can be pickled
    for worker processes; every run is independent (own environment, own log, own seed)."""
    cfg, task_path, condition, optimizer, seed, out_dir = job
    task, task_id = load_task(task_path), task_path.stem
    if condition == "S":
        res = run_scripted(cfg, task, task_id, seed, out_dir)
    else:
        res = run_baseline(cfg, task, task_id, condition, seed, out_dir, optimizer)
    metric = res["evaluation_metric"]
    return {
        "label": condition if optimizer is None else f"{condition}-{optimizer}",
        "condition": condition,
        "optimizer": res["optimizer"],
        "task_id": task_id,
        "seed": seed,
        "success": metric["success"],
        "evals_used": res["evals_used"],
        "first_success_eval": first_success_eval(
            cfg, task, out_dir / res["run_name"] / "events.jsonl"
        ),
        "budget": cfg.budget.max_evaluations,
        "final_error": final_error(metric),
        "wall_clock_s": round(res["wall_clock_s"], 3),
        "run_name": res["run_name"],
    }


def run_batch(
    cfg: ExperimentConfig,
    task_paths: list[Path],
    conditions: list[tuple[str, str | None]],
    seeds: list[int],
    out_dir: Path,
    jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run every (task, condition, seed) and return one result row per run, in that order.

    With jobs > 1 the runs go to worker processes; results are identical to a serial batch
    because runs share nothing and each is seeded by its own `seed`.
    """
    work = [
        (cfg, task_path, condition, optimizer, seed, out_dir)
        for task_path in task_paths
        for condition, optimizer in conditions
        for seed in seeds
    ]
    if jobs <= 1:
        return [_one_run(w) for w in work]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_one_run, work))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per condition label (see module docs for the definitions)."""
    out = []
    for label in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == label]
        k = sum(r["success"] for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        firsts = [r["first_success_eval"] for r in rs if r["first_success_eval"] is not None]
        by = {
            f"succeeded_by_{pct}pct": sum(
                r["first_success_eval"] is not None
                and r["first_success_eval"] <= math.ceil(pct / 100 * r["budget"])
                for r in rs
            )
            / len(rs)
            for pct in (25, 50, 100)
        }
        out.append(
            {
                "label": label,
                "runs": len(rs),
                "successes": k,
                "success_rate": k / len(rs),
                "wilson_low": lo,
                "wilson_high": hi,
                **by,
                "runs_with_a_success": len(firsts),
                "median_evals_to_first_success": statistics.median(firsts) if firsts else None,
                "mean_final_error": statistics.fmean(r["final_error"] for r in rs),
            }
        )
    return out


def format_summary(summary: list[dict[str, Any]]) -> str:
    """Markdown table of `summarize` output."""
    head = (
        "| condition | runs | success rate | Wilson 95% | by 25% | by 50% | by 100% | "
        "median evals to first success (n) | mean final error |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for s in summary:
        med = (
            "n/a (n=0)"
            if s["median_evals_to_first_success"] is None
            else f"{s['median_evals_to_first_success']:g} (n={s['runs_with_a_success']})"
        )
        lines.append(
            f"| {s['label']} | {s['runs']} | {s['successes']}/{s['runs']} = "
            f"{s['success_rate']:.2f} | [{s['wilson_low']:.2f}, {s['wilson_high']:.2f}] | "
            f"{s['succeeded_by_25pct']:.2f} | {s['succeeded_by_50pct']:.2f} | "
            f"{s['succeeded_by_100pct']:.2f} | {med} | {s['mean_final_error']:.3f} |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--tasks", nargs="+", required=True, help="task files (globs allowed)")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 0-4 or 1,5,9")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--budget", type=int, help="override budget.max_evaluations (recorded)")
    ap.add_argument("--csv", default="results.csv", help="results CSV (summary goes next to it)")
    args = ap.parse_args(argv)
    paths = sorted(Path(p) for pat in args.tasks for p in glob.glob(pat))
    cfg = load_config(args.config)
    if args.budget is not None:
        cfg = replace(cfg, budget=BudgetConfig(args.budget))
    rows = run_batch(
        cfg,
        paths,
        parse_conditions(args.conditions),
        parse_seeds(args.seeds),
        Path(args.out),
        args.jobs,
    )
    csv_path = Path(args.csv)
    write_csv(csv_path, rows, RESULT_FIELDS)
    summary = summarize(rows)
    write_csv(csv_path.with_name(csv_path.stem + "_summary.csv"), summary)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
