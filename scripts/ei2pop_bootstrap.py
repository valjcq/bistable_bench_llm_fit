"""Extra analysis of the ei_2pop test (same data, added after the results were known).

Per condition: tasks solved (>= 3 of 5 seeds) out of 9, and a 95% interval of the success rate
from a bootstrap over TASKS (resample the 9 tasks with replacement and keep each task's 5 seeds
together, so seeds of one task are not treated as independent). Percentile interval, 20000
resamples, fixed rng seed 0. Writes results/ei_test/bootstrap.md.

    python scripts/ei2pop_bootstrap.py
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["A", "B-cma", "B-tpe", "C-cma", "C-tpe"]
N_BOOT = 20000

rows = list(csv.DictReader(open(ROOT / "results/ei_test/test.csv")))
tasks = sorted({r["task_id"] for r in rows})
succ = {lab: collections.defaultdict(list) for lab in LABELS}
for r in rows:
    succ[r["label"]][r["task_id"]].append(r["success"] == "True")

rng = np.random.default_rng(0)
idx = rng.integers(0, len(tasks), size=(N_BOOT, len(tasks)))  # same resamples for every condition
intervals = {}
lines = [
    "| condition | success rate (pooled) | tasks solved (>= 3/5 seeds) | "
    "bootstrap-over-tasks 95% interval |",
    "|---|---|---|---|",
]
for lab in LABELS:
    per_task = np.array([np.mean(succ[lab][t]) for t in tasks])  # 5 seeds per task: equal weights
    solved = int((per_task >= 0.6).sum())
    boot = per_task[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    intervals[lab] = (lo, hi)
    lines.append(
        f"| {lab} | {per_task.mean():.3f} | {solved}/{len(tasks)} | [{lo:.2f}, {hi:.2f}] |"
    )
apart = [
    f"{a} vs {b}"
    for i, a in enumerate(LABELS)
    for b in LABELS[i + 1 :]
    if intervals[a][1] < intervals[b][0] or intervals[b][1] < intervals[a][0]
]
lines.append("")
lines.append("Non-overlapping bootstrap intervals (unrounded): " + ", ".join(apart))
text = "\n".join(lines) + "\n"
(ROOT / "results/ei_test/bootstrap.md").write_text(text)
print(text)
