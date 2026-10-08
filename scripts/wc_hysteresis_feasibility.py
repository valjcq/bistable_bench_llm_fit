"""EXPLORATORY feasibility of hysteresis tasks for Wilson-Cowan (nothing is adopted or planted).

Idea: instead of two stable states at one input P, ask for a P interval where the model has >= 2
stable states (a hysteresis loop in P, as in the paper's Figs. 5-8). This script only measures how
common that is: for N random draws in the wc1972 search box (Q = 0), sweep P over [-0.5, 1.0] in
steps of 0.01 and record the P interval(s) with >= 2 stable states.

Stable states are found as planting does (checker multi-start with 16 starts + the box corners,
no targets), not with the 256-start default; a comparison on a subset is printed at the end.

    python scripts/wc_hysteresis_feasibility.py [N]   # default 2000; writes
                                                       # results/wc_hysteresis_feasibility.md
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench.checker import find_stable_states
from bench.config import load_config
from bench.models.wc1972 import with_free_input

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/exp_wc1972.yaml")
MODEL = with_free_input(CFG.model(), -2.0, 2.0)  # P becomes a parameter we can sweep
PS = np.round(np.arange(-0.5, 1.0 + 1e-9, 0.01), 2)
TOL = CFG.task.tol
MIN_SEP = CFG.task.min_separation_tol_factor * TOL
CHEAP = replace(CFG.checker, n_starts=CFG.task.prefilter_starts)
FULL = CFG.checker
HIGH_LO, HIGH_HI = 0.05, 0.45  # the active-state window of the adopted task rule


def stable_E(params: dict[str, float], P: float, cfg=CHEAP) -> list[float]:
    stable = find_stable_states(MODEL, {**params, "P": float(P)}, cfg)
    return sorted(fp.observed[0] for fp in stable)


def intervals_of(params: dict[str, float], cfg=CHEAP) -> list[tuple[float, float]]:
    """Maximal runs of consecutive grid P values with >= 2 stable states: (first P, last P)."""
    out: list[tuple[float, float]] = []
    run_start = prev = None
    for P in PS:
        if len(stable_E(params, P, cfg)) >= 2:
            if run_start is None:
                run_start = P
            prev = P
        elif run_start is not None:
            out.append((float(run_start), float(prev)))
            run_start = None
    if run_start is not None:
        out.append((float(run_start), float(prev)))
    return out


def analyse(params: dict[str, float]) -> list[dict]:
    """Per bistable interval: bounds, width, and the stable states at the midpoint."""
    rows = []
    for a, b in intervals_of(params):
        mid = round((a + b) / 2, 3)
        E = stable_E(params, mid)
        rows.append({"lo": a, "hi": b, "width": round(b - a, 2), "mid": mid, "E_mid": E})
    return rows


def main(n: int = 2000) -> str:
    rng = np.random.default_rng(0)
    draws = [MODEL.space.from_unit(rng.random(MODEL.space.dim)) for _ in range(n)]
    draws = [{k: v for k, v in d.items() if k != "P"} for d in draws]  # P is swept, not drawn
    with ProcessPoolExecutor() as pool:
        per_draw = list(pool.map(analyse, draws, chunksize=10))

    with_interval = [r for r in per_draw if r]
    all_iv = [iv for r in per_draw for iv in r]
    widths = np.array([iv["width"] for iv in all_iv])
    widest = np.array([max(iv["width"] for iv in r) for r in with_interval])

    def qualifies(iv: dict) -> bool:  # inside [-0.3, 0.6], width in [0.1, 0.5]
        return iv["lo"] >= -0.3 and iv["hi"] <= 0.6 and 0.1 <= iv["width"] <= 0.5

    good_draws = [r for r in per_draw if any(qualifies(iv) for iv in r)]
    good_iv = [iv for iv in all_iv if qualifies(iv)]

    def active_in_window(iv: dict) -> bool:
        return bool(iv["E_mid"]) and HIGH_LO <= iv["E_mid"][-1] <= HIGH_HI

    def usable(iv: dict) -> bool:  # active state in window AND >= 3 tol above the lowest state
        return active_in_window(iv) and iv["E_mid"][-1] - iv["E_mid"][0] >= MIN_SEP

    q = [5, 25, 50, 75, 95]
    pct = lambda x: np.round(np.percentile(x, q), 2).tolist()  # noqa: E731
    lines = [
        "EXPLORATORY: hysteresis feasibility for Wilson-Cowan (not adopted, no tasks planted).",
        "",
        f"{n} random draws in the wc1972 box (seed 0), Q = 0; P swept over [-0.5, 1.0] in steps of "
        "0.01; stable states found with the planting-level checker (16 starts + corners). An "
        "interval = maximal run of consecutive P values with >= 2 stable states; width = last P "
        "- first P.",
        "",
        f"- draws with at least one bistable interval: {len(with_interval)}/{n} = "
        f"{len(with_interval) / n:.3f}",
        f"- number of intervals found: {len(all_iv)} (draws with 2 or more intervals: "
        f"{sum(len(r) >= 2 for r in per_draw)})",
        f"- interval width, all intervals, percentiles {q}: {pct(widths)}; mean "
        f"{widths.mean():.2f}; widest interval per draw: {pct(widest)}",
        "- draws with an interval inside [-0.3, 0.6] and width in [0.1, 0.5]: "
        f"{len(good_draws)}/{n} = {len(good_draws) / n:.3f} ({len(good_iv)} such intervals)",
        "",
        f"Active state at the interval midpoint (highest stable E there; window [{HIGH_LO}, "
        f"{HIGH_HI}]):",
        "",
        "| intervals | count | active E in window | in window and >= 3 tol above the lowest |",
        "|---|---|---|---|",
        f"| all | {len(all_iv)} | {sum(map(active_in_window, all_iv))} "
        f"({np.mean([active_in_window(iv) for iv in all_iv]):.3f}) | {sum(map(usable, all_iv))} "
        f"({np.mean([usable(iv) for iv in all_iv]):.3f}) |",
        f"| inside [-0.3, 0.6], width in [0.1, 0.5] | {len(good_iv)} | "
        f"{sum(map(active_in_window, good_iv))} "
        f"({np.mean([active_in_window(iv) for iv in good_iv]) if good_iv else float('nan'):.3f}) |"
        f" {sum(map(usable, good_iv))} "
        f"({np.mean([usable(iv) for iv in good_iv]) if good_iv else float('nan'):.3f}) |",
    ]
    draws_with_usable_good = sum(any(qualifies(iv) and usable(iv) for iv in r) for r in per_draw)
    lines += [
        "",
        f"- draws with a qualifying interval whose midpoint has a usable active state: "
        f"{draws_with_usable_good}/{n} = {draws_with_usable_good / n:.4f}",
    ]
    # Effort check on 100 draws: does the 256-start search find more bistable P values?
    more = 0
    for d in draws[:100]:
        for P in PS[::10]:
            more += len(stable_E(d, P, FULL)) >= 2 and len(stable_E(d, P, CHEAP)) < 2
    lines += [
        "",
        f"Effort check: on the first 100 draws, at every 10th P value the 256-start search found "
        f"bistability where the 16-start search did not in {more} of {100 * len(PS[::10])} cases.",
    ]
    text = "\n".join(lines) + "\n"
    (ROOT / "results/wc_hysteresis_feasibility.md").write_text(text)
    return text


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000))
