"""Feasibility report for Wilson-Cowan task planting (no task files are produced).

Samples N parameter sets in the wc1972 search box, each with a task input P drawn as at
planting (uniform in [-0.3, 0.3], |P| >= 0.05), and measures how often a random draw has the
stable states a task needs. Two search effort levels of the checker are compared:
  prefilter  16 multi-start Newton runs: what `make_task` uses to screen draws
  full       256 runs: the checker's default
"Usable" stable states have observed E inside the planting window (5-95% of the reference range
[0, 0.5]); K usable states must also be pairwise separated by >= 3 tol (tol = 0.01).

    python scripts/wc_feasibility.py [N]      # default N = 5000; writes results/wc_feasibility.md
"""

from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench.batch import wilson_interval
from bench.checker import find_stable_states
from bench.config import load_config
from bench.tasks import draw_task_inputs, target_window_bounds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/exp_wc1972.yaml")
MODEL = CFG.model()
TOL = CFG.task.tol
MIN_SEP = CFG.task.min_separation_tol_factor * TOL
LO, HI = (float(x[0]) for x in target_window_bounds(MODEL, CFG.task.target_window))


def max_separated(values: list[float]) -> int:
    """Largest number of values pairwise >= MIN_SEP apart (greedy on sorted values is optimal)."""
    count, last = 0, -math.inf
    for v in sorted(values):
        if v - last >= MIN_SEP:
            count, last = count + 1, v
    return count


def classify(job: tuple[dict[str, float], dict[str, float]]) -> dict[str, tuple]:
    """For one draw and each effort level: (n stable, n usable, max separated usable); also the
    sorted observed E of the stable states found at the full level, under key "E"."""
    params, inputs = job
    model = MODEL.with_inputs(inputs)
    out: dict[str, tuple] = {}
    for name, starts in (("prefilter", CFG.task.prefilter_starts), ("full", CFG.checker.n_starts)):
        stable = find_stable_states(model, params, replace(CFG.checker, n_starts=starts))
        usable = [fp.observed[0] for fp in stable if LO <= fp.observed[0] <= HI]
        out[name] = (len(stable), len(usable), max_separated(usable))
        if name == "full":
            out["E"] = tuple(sorted(fp.observed[0] for fp in stable))
    return out


def main(n: int = 5000) -> str:
    rng = np.random.default_rng(0)
    jobs = []
    for _ in range(n):  # same draw order as make_task: parameters, then the task input
        params = MODEL.space.from_unit(rng.random(MODEL.space.dim))
        jobs.append((params, draw_task_inputs(MODEL, rng)))
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(classify, jobs, chunksize=25))

    lines = [
        f"Wilson-Cowan feasibility: {n} draws in the wc1972 box (seed 0), P ~ U[-0.3, 0.3], "
        f"|P| >= 0.05; tol = {TOL}, planting window E in [{LO:.3f}, {HI:.3f}], "
        f"separation >= {MIN_SEP:.2f}.",
        "",
        "| criterion | prefilter (16 starts): fraction [Wilson 95%] | full (256 starts): fraction "
        "[Wilson 95%] |",
        "|---|---|---|",
    ]
    rows = {
        ">= 2 stable states": lambda r: r[0] >= 2,
        ">= 3 stable states": lambda r: r[0] >= 3,
        ">= 2 usable stable states (E in window)": lambda r: r[1] >= 2,
        "K=2: >= 2 usable, separated by >= 3 tol": lambda r: r[2] >= 2,
        "K=3: >= 3 usable, separated by >= 3 tol": lambda r: r[2] >= 3,
    }
    frac: dict[tuple[str, str], tuple[int, float, tuple[float, float]]] = {}
    for label, test in rows.items():
        cells = []
        for level in ("prefilter", "full"):
            k = sum(test(r[level]) for r in results)
            lo, hi = wilson_interval(k, n)
            frac[(label, level)] = (k, k / n, (lo, hi))
            cells.append(f"{k}/{n} = {k / n:.4f} [{lo:.4f}, {hi:.4f}]")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    lines += ["", "Draws per task (1 / fraction; range from the Wilson interval):", ""]
    lines += ["| K | prefilter (what planting uses) | full |", "|---|---|---|"]
    for K in (2, 3):
        label = f"K={K}: >= {K} usable, separated by >= 3 tol"
        cells = []
        for level in ("prefilter", "full"):
            k, f, (lo, hi) = frac[(label, level)]
            cells.append(
                "no success in the sample"
                if k == 0
                else f"{1 / f:.0f} [{1 / hi:.0f}, " + ("inf" if lo == 0 else f"{1 / lo:.0f}") + "]"
            )
        lines.append(f"| {K} | {cells[0]} | {cells[1]} |")
    # Where do the stable states sit? (draws with >= 2 stable states, full effort)
    multi = [r["E"] for r in results if len(r["E"]) >= 2]
    lows, highs = np.array([e[0] for e in multi]), np.array([e[-1] for e in multi])
    q = [5, 25, 50, 75, 95]
    lines += [
        "",
        f"Where the stable states sit ({len(multi)} draws with >= 2 stable states, full effort):",
        "",
        f"- lowest stable E, percentiles {q}: {np.round(np.percentile(lows, q), 4).tolist()}",
        f"- highest stable E, percentiles {q}: {np.round(np.percentile(highs, q), 4).tolist()}",
        f"- lowest state inside the window (E >= {LO:.3f}): {np.mean(lows >= LO):.4f} of them; "
        f"highest inside (E <= {HI:.3f}): {np.mean(highs <= HI):.4f}; both: "
        f"{np.mean((lows >= LO) & (highs <= HI)):.4f}",
        "",
        "Sensitivity of the K=2 count (of all draws) to the window, same separation rule:",
        "",
        "| window for E | draws with K=2 |",
        "|---|---|",
    ]
    for lo_w, hi_w in ((LO, HI), (0.005, 0.495), (0.0, 0.5)):
        k2 = sum(max_separated([x for x in e if lo_w <= x <= hi_w]) >= 2 for e in multi)
        lines.append(f"| [{lo_w:.3f}, {hi_w:.3f}] | {k2}/{n} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results/wc_feasibility.md").write_text(text)
    return text


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000))
