"""Task-level analysis of a test run: tasks solved and a bootstrap over tasks (generic version of
scripts/ei2pop_bootstrap.py; same method, any results CSV from `bench.batch`).

Per condition: pooled success rate, tasks solved (>= 3 of 5 seeds, i.e. a rate >= 0.6), and a
percentile 95% interval from 20000 bootstrap resamples of the TASKS (each task's seeds kept
together; the same resamples for every condition, rng seed 0). Also lists the pairs of conditions
whose bootstrap intervals do not overlap.

    python scripts/task_bootstrap.py results/wc_test/test.csv results/wc_test/bootstrap.md
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

import numpy as np

N_BOOT = 20000
ORDER = ["A", "B-cma", "B-tpe", "C-cma", "C-tpe"]


def analyse(csv_path: str | Path) -> tuple[str, dict[str, tuple[float, float]]]:
    rows = list(csv.DictReader(open(csv_path)))
    labels = [x for x in ORDER if any(r["label"] == x for r in rows)]
    tasks = sorted({r["task_id"] for r in rows})
    succ: dict[str, dict[str, list[bool]]] = {lab: collections.defaultdict(list) for lab in labels}
    for r in rows:
        succ[r["label"]][r["task_id"]].append(r["success"] == "True")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(tasks), size=(N_BOOT, len(tasks)))
    lines = [
        "| condition | success rate (pooled) | tasks solved (>= 3/5 seeds) | "
        "bootstrap-over-tasks 95% interval |",
        "|---|---|---|---|",
    ]
    intervals: dict[str, tuple[float, float]] = {}
    for lab in labels:
        per_task = np.array([np.mean(succ[lab][t]) for t in tasks])
        solved = int((per_task >= 0.6).sum())
        lo, hi = np.percentile(per_task[idx].mean(axis=1), [2.5, 97.5])
        intervals[lab] = (float(lo), float(hi))
        lines.append(
            f"| {lab} | {per_task.mean():.3f} | {solved}/{len(tasks)} | [{lo:.2f}, {hi:.2f}] |"
        )
    apart = [
        f"{a} vs {b}"
        for i, a in enumerate(labels)
        for b in labels[i + 1 :]
        if intervals[a][1] < intervals[b][0] or intervals[b][1] < intervals[a][0]
    ]
    lines += [
        "",
        "Non-overlapping bootstrap intervals (unrounded): " + ", ".join(apart or ["none"]),
    ]
    return "\n".join(lines) + "\n", intervals


if __name__ == "__main__":
    text, _ = analyse(sys.argv[1])
    Path(sys.argv[2]).write_text(text)
    print(text)
