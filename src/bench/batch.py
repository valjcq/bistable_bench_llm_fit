"""Batch runner and summary: conditions x tasks x seeds -> one CSV of results + a summary table.

    python -m bench.batch CONFIG --tasks tasks/toy_*.json --conditions A B:cma B:tpe C:cma C:tpe
        --seeds 0-4 [--out runs/] [--csv results.csv]

A condition is `A`, or `B:<optimizer>` / `C:<optimizer>`; the label in the tables is `B-cma`.

How the summary numbers are defined
- success rate: fraction of runs whose SUBMITTED parameters pass the final check
  (evaluation_metric.success), with a Wilson 95% interval.
- evals to first success: the index (1-based) of the first evaluation, in call order, whose
  parameters pass the checker; the median is over the runs that have one. It is computed after
  the fact from events.jsonl, not by the optimizer, and the checks done for it are NOT counted
  against any budget (they are analysis, not part of the episode). To avoid checking every point:
  for the expert objective, a score <= tol is equivalent to success (matched targets add at most
  tol each, an unmatched one adds >= penalty/K > tol; test_scoring), so no checker call is needed;
  for the naive objective, which can be fooled, only evaluations with score <= tol are candidates
  and each candidate is verified with the checker. A naive-objective success with score > tol is
  therefore not credited (conservative).
  If `forbid_extra_stable_states` is on, expert candidates are verified too.
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
from pathlib import Path
from typing import Any

from bench.checker import check
from bench.config import ExperimentConfig, load_config
from bench.logging import read_events
from bench.runner import CONDITIONS, run_baseline
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
    """1-based index of the first evaluated point that passes the checker (see module docs)."""
    model = cfg.model()
    tol = cfg.task.tol
    evals = [e for e in read_events(events_path) if e["name"] == "evaluate"]
    for i, e in enumerate(evals, start=1):
        if e["output"]["value"] > tol:
            continue  # not a candidate (cannot be an expert success; not credited if naive)
        if e["arguments"]["objective"] == "expert" and not cfg.task.forbid_extra_stable_states:
            return i
        if check(model, e["arguments"]["params"], task.targets, cfg.checker).success:
            return i
    return None


def parse_conditions(specs: Iterable[str]) -> list[tuple[str, str | None]]:
    """['A', 'B:cma'] -> [('A', None), ('B', 'cma')]."""
    out = []
    for s in specs:
        cond, _, opt = s.partition(":")
        if cond not in CONDITIONS or (cond == "A") != (opt == ""):
            raise ValueError(f"bad condition {s!r}: use A, B:<optimizer> or C:<optimizer>")
        out.append((cond, opt or None))
    return out


def parse_seeds(text: str) -> list[int]:
    """'0-4' -> [0..4]; '1,5,9' -> [1, 5, 9]."""
    seeds: list[int] = []
    for part in text.split(","):
        lo, dash, hi = part.partition("-")
        seeds += range(int(lo), int(hi) + 1) if dash else [int(lo)]
    return seeds


def run_batch(
    cfg: ExperimentConfig,
    task_paths: list[Path],
    conditions: list[tuple[str, str | None]],
    seeds: list[int],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Run every (condition, task, seed) and return one result row per run."""
    rows = []
    for task_path in task_paths:
        task, task_id = load_task(task_path), task_path.stem
        for condition, optimizer in conditions:
            for seed in seeds:
                res = run_baseline(cfg, task, task_id, condition, seed, out_dir, optimizer)
                events = out_dir / res["run_name"] / "events.jsonl"
                metric = res["evaluation_metric"]
                rows.append(
                    {
                        "label": condition if optimizer is None else f"{condition}-{optimizer}",
                        "condition": condition,
                        "optimizer": res["optimizer"],
                        "task_id": task_id,
                        "seed": seed,
                        "success": metric["success"],
                        "evals_used": res["evals_used"],
                        "first_success_eval": first_success_eval(cfg, task, events),
                        "final_error": final_error(metric),
                        "wall_clock_s": round(res["wall_clock_s"], 3),
                        "run_name": res["run_name"],
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per condition label (see module docs for the definitions)."""
    out = []
    for label in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == label]
        k = sum(r["success"] for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        firsts = [r["first_success_eval"] for r in rs if r["first_success_eval"] is not None]
        out.append(
            {
                "label": label,
                "runs": len(rs),
                "successes": k,
                "success_rate": k / len(rs),
                "wilson_low": lo,
                "wilson_high": hi,
                "runs_with_a_success": len(firsts),
                "median_evals_to_first_success": statistics.median(firsts) if firsts else None,
                "mean_final_error": statistics.fmean(r["final_error"] for r in rs),
            }
        )
    return out


def format_summary(summary: list[dict[str, Any]]) -> str:
    """Markdown table of `summarize` output."""
    head = (
        "| condition | runs | success rate | Wilson 95% | runs with a success | "
        "median evals to first success | mean final error |\n|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for s in summary:
        med = (
            "n/a"
            if s["median_evals_to_first_success"] is None
            else (f"{s['median_evals_to_first_success']:g}")
        )
        lines.append(
            f"| {s['label']} | {s['runs']} | {s['successes']}/{s['runs']} = "
            f"{s['success_rate']:.2f} | [{s['wilson_low']:.2f}, {s['wilson_high']:.2f}] | "
            f"{s['runs_with_a_success']} | {med} | {s['mean_final_error']:.3f} |"
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
    ap.add_argument("--csv", default="results.csv", help="results CSV (summary goes next to it)")
    args = ap.parse_args(argv)
    paths = sorted(Path(p) for pat in args.tasks for p in glob.glob(pat))
    rows = run_batch(
        load_config(args.config),
        paths,
        parse_conditions(args.conditions),
        parse_seeds(args.seeds),
        Path(args.out),
    )
    csv_path = Path(args.csv)
    write_csv(csv_path, rows, RESULT_FIELDS)
    summary = summarize(rows)
    write_csv(csv_path.with_name(csv_path.stem + "_summary.csv"), summary)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
