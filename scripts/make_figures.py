"""The README figure, drawn from committed result files only (no run directories needed).

Two panels, the two findings of the project:
  left   Wilson-Cowan held-out test (budget 200, 45 runs per condition): success by loss and
         optimizer, from results/wc_test/test_summary.csv. Shows that the loss, not the
         optimizer, makes the difference.
  right  invented solutions on provably impossible tasks, over every agent run (first run, v2 and
         the extension): the agent with and without the v2 prompt sentence, against pipeline P.
         Counted from the per-episode CSVs in results/agent_eval*/; harness errors are excluded,
         as in the reports.
Error bars are Wilson 95% intervals. matplotlib is not a project dependency, so run it with

    uv run --with matplotlib python scripts/make_figures.py   # writes docs/figures/summary.png
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures" / "summary.png"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EXPERT, NAIVE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def wc_test() -> list[tuple[str, int, int, str]]:
    """(label, successes, runs, colour) per condition of the Wilson-Cowan test."""
    names = {
        "A": ("random search", NEUTRAL),
        "B-cma": ("naive + CMA-ES", NAIVE),
        "B-tpe": ("naive + TPE", NAIVE),
        "C-cma": ("expert v2 + CMA-ES", EXPERT),
        "C-tpe": ("expert v2 + TPE", EXPERT),
    }
    rows = csv.DictReader(open(ROOT / "results/wc_test/test_summary.csv"))
    out = []
    for r in rows:
        label, colour = names[r["label"]]
        out.append((label, int(r["successes"]), int(r["runs"]), colour))
    return out


def invented(paths: list[Path]) -> tuple[int, int]:
    """Invented solutions and episodes on infeasible tasks, harness errors excluded."""
    k = n = 0
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["feasible"] != "False" or r["harness_error"] == "True":
                continue
            n += 1
            k += r["category"] == "invented_solution"
    return k, n


def honesty() -> list[tuple[str, int, int, str]]:
    res = ROOT / "results"
    v1 = [p for p in (res / "agent_eval").glob("*.csv") if not p.name.startswith("reference")]
    v1 += sorted((res / "agent_eval_ext").glob("v1_*.csv"))
    v2 = sorted((res / "agent_eval_v2").glob("*.csv")) + sorted(
        (res / "agent_eval_ext").glob("v2_*.csv")
    )
    ref = sorted((res / "agent_eval").glob("reference_*.csv")) + sorted(
        (res / "agent_eval_ext").glob("reference_*.csv")
    )
    return [
        ("agent, prompt v1", *invented(v1), EXPERT),
        ("agent, prompt v2\n(+ sentence on scores)", *invented(v2), EXPERT),
        ("fixed pipeline", *invented(ref), NEUTRAL),
    ]


def panel(ax, data, title: str, xlabel: str, xmax: float = 1.0) -> None:
    ys = list(range(len(data)))[::-1]
    for y, (_label, k, n, colour) in zip(ys, data, strict=True):
        lo, hi = wilson(k, n)
        ax.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
        ax.plot(k / n, y, "o", ms=9, color=colour, mec="white", mew=2)
        ax.text(1.03, y, f"{k}/{n}", va="center", ha="left", color=MUTED, fontsize=9,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(ys, [d[0] for d in data], color=INK, fontsize=9.5)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(-0.6, len(data) - 0.4)


def main() -> None:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [5, 3], "wspace": 0.75}
    )
    panel(a, wc_test(), "The loss matters more than the optimizer",
          "Wilson-Cowan held-out test, success rate (budget 200)")
    panel(b, honesty(), "The agent invents solutions",
          "claimed a solution on an impossible task", xmax=0.4)
    fig.text(0.01, -0.08, "Dots: rate. Bars: Wilson 95% interval. Right: every agent run pooled.",
             color=MUTED, fontsize=8)
    fig.patch.set_facecolor("white")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for label, k, n, _ in honesty():
        print(f"  {label.splitlines()[0]}: {k}/{n}")


if __name__ == "__main__":
    main()
