"""Phase 0 diagnosis from existing logs (no new optimizer runs): why does C do worse than B?

Reads runs/phase0/grid (toy grid, budget 200) and the task files; every check below is an
off-budget analysis call of the checker on points that the runs already evaluated.
Writes results/phase0/diagnosis.md (the numbers quoted in docs/phase0.md).

    python scripts/phase0_diagnosis.py
"""

from __future__ import annotations

import collections
import csv
import statistics
from pathlib import Path

import numpy as np

from bench.checker import check
from bench.config import load_config
from bench.logging import read_events
from bench.tasks import load_task

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "runs/phase0/grid"
CFG = load_config(ROOT / "configs/exp_toy_1pop.yaml")
MODEL = CFG.model()
TASKS = {i: load_task(ROOT / f"tasks/toy_1pop_K2_seed{i}.json") for i in range(5)}
LABELS = ("B-cma", "C-cma", "C-tpe")
out: list[str] = []


def runs(label: str):
    """(task index, seed, result dict, evaluate events) for every grid run of a condition."""
    for r in csv.DictReader(open(ROOT / "results/phase0/toy_grid.csv")):
        if r["label"] == label:
            d = GRID / r["run_name"]
            ev = [e for e in read_events(d / "events.jsonl") if e["name"] == "evaluate"]
            yield int(r["task_id"][-1]), int(r["seed"]), r["success"] == "True", ev


def verdict(task_idx: int, params: dict):
    return check(MODEL, params, TASKS[task_idx].targets, CFG.checker)


# ---- (a) final best expert score in failed C runs -----------------------------------------
bins = [(0, 0.02, "< tol"), (0.02, 0.5, "[tol, 0.5)"), (0.5, 0.6, "[0.5, 0.6)"),
        (0.6, 0.75, "[0.6, 0.75)"), (0.75, 1.0, "[0.75, 1.0)"), (1.0, 9, ">= 1.0")]  # fmt: skip
out.append("## (a) Final best expert score of FAILED C runs\n")
out.append("| condition | failed runs | " + " | ".join(b[2] for b in bins) + " |")
out.append("|---|---|" + "---|" * len(bins))
detail = {}
for label in ("C-cma", "C-tpe"):
    scores, matched = [], collections.Counter()
    for t, _s, ok, ev in runs(label):
        if ok:
            continue
        vals = [e["output"]["value"] for e in ev]
        best = ev[int(np.argmin(vals))]
        scores.append(min(vals))
        res = verdict(t, best["arguments"]["params"])
        matched[sum(res.target_matched)] += 1
    counts = [sum(lo <= x < hi for x in scores) for lo, hi, _ in bins]
    out.append(f"| {label} | {len(scores)} | " + " | ".join(map(str, counts)) + " |")
    detail[label] = (scores, matched)
for label, (scores, matched) in detail.items():
    out.append(
        f"\n{label}: median {statistics.median(scores):.3f}, min {min(scores):.3f}, "
        f"max {max(scores):.3f}; targets matched at the best point (checker): "
        f"{dict(sorted(matched.items()))} (key = number of targets matched)."
    )

# A monostable state sitting exactly on one target matches it (error 0); the other target is then
# at distance `separation` from the only fixed point, so the score is (0 + 1 + separation) / 2.
out.append("\n### Plateau level (1 + separation) / 2\n")
for label in ("C-cma", "C-tpe"):
    near, total = [], 0
    for t, _s, ok, ev in runs(label):
        if ok:
            continue
        total += 1
        sep = float(np.ptp(TASKS[t].targets))
        best = min(e["output"]["value"] for e in ev)
        near.append(abs(best - (1 + sep) / 2) < 0.01)
    out.append(f"- {label}: {sum(near)} of {total} failed runs end within 0.01 of their task's "
               "(1 + separation) / 2.")  # fmt: skip

# ---- (b) fraction of evaluations by number of stable states --------------------------------
out.append("\n## (b) What the evaluated points are (checker, off-budget), 25 runs x 200 evals\n")
out.append("| condition | monostable (1 stable state) | no stable state | bistable+ (>=2 stable) | "
           "success |\n|---|---|---|---|---|")  # fmt: skip
frac = {}
for label in ("B-cma", "C-cma"):
    c = collections.Counter()
    for t, _s, _ok, ev in runs(label):
        for e in ev:
            r = verdict(t, e["arguments"]["params"])
            c["mono" if r.n_stable == 1 else "none" if r.n_stable == 0 else "bi"] += 1
            c["succ"] += r.success
    n = sum(c[k] for k in ("mono", "none", "bi"))
    frac[label] = c
    out.append(f"| {label} | {c['mono'] / n:.3f} | {c['none'] / n:.3f} | {c['bi'] / n:.3f} | "
               f"{c['succ'] / n:.3f} |")  # fmt: skip

# ---- (c) example trajectories of stuck C-cma runs ------------------------------------------
out.append("\n## (c) Best-so-far expert score of stuck C-cma runs\n")
marks = [1, 10, 25, 50, 100, 150, 200]
failed = [(t, s, ev) for t, s, ok, ev in runs("C-cma") if not ok]
picks = [x for x in failed if (x[0], x[1]) in ((3, 0), (0, 0))] or failed[:2]
out.append(
    "| run | " + " | ".join(f"@{m}" for m in marks) + " | last improvement | "
    "monostable evals | evals with score >= 0.5 |\n|---|" + "---|" * (len(marks) + 3)
)
for t, s, ev in picks:
    vals = np.array([e["output"]["value"] for e in ev])
    best = np.minimum.accumulate(vals)
    last = (
        int(np.nonzero(np.diff(best) < -1e-6)[0].max()) + 2 if (np.diff(best) < -1e-6).any() else 1
    )
    mono = sum(verdict(t, e["arguments"]["params"]).n_stable == 1 for e in ev)
    out.append(f"| task{t} seed{s} | " + " | ".join(f"{best[m - 1]:.3f}" for m in marks)
               + f" | {last} | {mono}/200 | {int((vals >= 0.5).sum())}/200 |")  # fmt: skip

# ---- 3. per-task descriptors ---------------------------------------------------------------
out.append("\n## Per-task descriptors and success (of 5 seeds)\n")
lo_w, hi_w = CFG.task.target_window
succ = collections.defaultdict(lambda: collections.defaultdict(int))
for r in csv.DictReader(open(ROOT / "results/phase0/toy_grid.csv")):
    succ[int(r["task_id"][-1])][r["label"]] += r["success"] == "True"
cols = ["A", "B-cma", "B-tpe", "C-cma", "C-tpe", "S"]
out.append("| task | low target | high target | low-edge dist | high-edge dist | separation | "
           "planted margin_eig | planted margin_dist | " + " | ".join(cols)
           + " |\n|---|---|---|---|---|---|---|---|" + "---|" * len(cols))  # fmt: skip
for i, task in TASKS.items():
    lo, hi = sorted(task.targets.ravel())
    res = check(MODEL, task.planted_params, task.targets, CFG.checker)
    out.append(
        f"| seed{i} | {lo:.3f} | {hi:.3f} | {lo - lo_w:.3f} | {hi_w - hi:.3f} | {hi - lo:.3f} | "
        f"{res.margin_eig:.4f} | {res.margin_dist:.3f} | "
        + " | ".join(str(succ[i][c]) for c in cols)
        + " |"
    )
out.append(
    "\nedge dist = distance of the target to the window edge (0.05 / 0.95); separation = "
    "high - low (the guard is 3 tol = 0.06); margins are those of the planted solution "
    "(`margin_eig` = slowest decay rate, `margin_dist` = distance to the unstable point)."
)

text = "\n".join(out) + "\n"
(ROOT / "results/phase0/diagnosis.md").write_text(text)
print(text)
