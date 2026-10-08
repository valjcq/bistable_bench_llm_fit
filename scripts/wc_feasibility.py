"""Feasibility report for Wilson-Cowan task planting (no task files are produced).

Samples N parameter sets in the wc1972 search box, each with a task input P drawn as at
planting (uniform in [-0.3, 0.3], |P| >= 0.05), and measures how often a random draw gives a
K = 2 task under a given window rule. Stable states are found as planting does (checker
multi-start with 16 starts + the box corners; the 256-start default is shown too).

Window rules (tol = 0.01, separation >= 3 tol = 0.03):
  asymmetric (the rule in configs/exp_wc1972.yaml): the HIGH target has E in [0.05, 0.45]; the LOW
              target is any stable state at least 3 tol below it (no window).
  symmetric   both targets have E in [0.025, 0.475] (5-95% of the reference range), the old rule.
Scenarios: the box of configs/wc1972.yaml, and (comparison only, not adopted) the same box with
a_e restricted to [1, 2].

    python scripts/wc_feasibility.py [N]      # default N = 5000; writes results/wc_feasibility.md
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench.batch import wilson_interval
from bench.checker import find_stable_states
from bench.config import load_config
from bench.models.base import ParamSpace, ParamSpec
from bench.tasks import draw_task_inputs, target_window_bounds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/exp_wc1972.yaml")
MODEL = CFG.model()
TOL = CFG.task.tol
MIN_SEP = CFG.task.min_separation_tol_factor * TOL
HIGH_LO, HIGH_HI = (float(x[0]) for x in target_window_bounds(MODEL, CFG.task.target_window))
SYM_LO, SYM_HI = (float(x[0]) for x in target_window_bounds(MODEL, (0.05, 0.95)))


def restrict(model, name: str, low: float, high: float):
    """The model with one free parameter's bounds narrowed (comparison scenario)."""
    free = tuple(
        ParamSpec(p.name, low, high, p.log) if p.name == name else p for p in model.space.free
    )
    return replace(model, space=ParamSpace(free, dict(model.space.fixed)))


SCENARIOS = {
    "box of wc1972.yaml (adopted)": MODEL,
    "a_e in [1, 2] (comparison only, not adopted)": restrict(MODEL, "a_e", 1.0, 2.0),
}


def asymmetric_ok(E: list[float]) -> bool:
    """A high target in [HIGH_LO, HIGH_HI] with a stable state >= 3 tol below it."""
    return any(HIGH_LO <= b <= HIGH_HI and any(b - a >= MIN_SEP for a in E) for b in E)


def symmetric_ok(E: list[float]) -> bool:
    """Two stable states inside [SYM_LO, SYM_HI], at least 3 tol apart."""
    inside = sorted(v for v in E if SYM_LO <= v <= SYM_HI)
    return any(b - a >= MIN_SEP for a in inside for b in inside)


def classify(job) -> dict[str, tuple[list[float], bool, bool]]:
    """For one draw and each effort level: (sorted stable E, asymmetric_ok, symmetric_ok)."""
    scenario, params, inputs = job  # the model itself is not picklable: look it up by name
    task_model = SCENARIOS[scenario].with_inputs(inputs)
    out = {}
    for level, starts in (("prefilter", CFG.task.prefilter_starts), ("full", CFG.checker.n_starts)):
        stable = find_stable_states(task_model, params, replace(CFG.checker, n_starts=starts))
        E = sorted(fp.observed[0] for fp in stable)
        out[level] = (E, asymmetric_ok(E), symmetric_ok(E))
    return out


def fmt(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.4f} [{lo:.4f}, {hi:.4f}]"


def draws(k: int, n: int) -> str:
    if k == 0:
        return "no success in the sample"
    lo, hi = wilson_interval(k, n)
    return f"{n / k:.0f} [{1 / hi:.0f}, {'inf' if lo == 0 else f'{1 / lo:.0f}'}]"


def main(n: int = 5000) -> str:
    lines = [
        f"Wilson-Cowan feasibility, K=2: {n} draws per scenario (seed 0), P ~ U[-0.3, 0.3], "
        f"|P| >= 0.05, tol = {TOL}.",
        f"Asymmetric rule: high target E in [{HIGH_LO:.2f}, {HIGH_HI:.2f}], low target free, "
        f"separation >= {MIN_SEP:.2f}. Symmetric (old) rule: both in [{SYM_LO:.3f}, {SYM_HI:.3f}].",
    ]
    for scenario, model in SCENARIOS.items():
        rng = np.random.default_rng(0)
        jobs = []
        for _ in range(n):  # same draw order as make_task: parameters, then the task input
            params = model.space.from_unit(rng.random(model.space.dim))
            jobs.append((scenario, params, draw_task_inputs(model, rng)))
        with ProcessPoolExecutor() as pool:
            res = list(pool.map(classify, jobs, chunksize=25))
        lines += ["", f"### {scenario}", ""]
        lines += ["| criterion | prefilter (planting) | full (256 starts) |", "|---|---|---|"]
        rows = {
            ">= 2 stable states": lambda r: len(r[0]) >= 2,
            "K=2, asymmetric rule (adopted)": lambda r: r[1],
            "K=2, symmetric rule (old)": lambda r: r[2],
        }
        counts = {}
        for label, test in rows.items():
            ks = {lv: sum(test(r[lv]) for r in res) for lv in ("prefilter", "full")}
            counts[label] = ks
            lines.append(f"| {label} | {fmt(ks['prefilter'], n)} | {fmt(ks['full'], n)} |")
        ks = counts["K=2, asymmetric rule (adopted)"]
        lines += [
            "",
            f"Draws per task, K=2, asymmetric rule: **{draws(ks['prefilter'], n)}** with the "
            f"planting prefilter, {draws(ks['full'], n)} with the full search "
            "(1 / fraction, [range from the Wilson interval]).",
        ]
        multi = [r["full"][0] for r in res if len(r["full"][0]) >= 2]
        lows = np.array([e[0] for e in multi])
        highs = np.array([e[-1] for e in multi])
        q = [5, 25, 50, 75, 95]
        lines += [
            "",
            f"Stable states of the {len(multi)} draws with >= 2 (full): lowest E percentiles {q}: "
            f"{np.round(np.percentile(lows, q), 4).tolist()}; highest E: "
            f"{np.round(np.percentile(highs, q), 4).tolist()}.",
        ]
    text = "\n".join(lines) + "\n"
    (ROOT / "results/wc_feasibility.md").write_text(text)
    return text


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000))
