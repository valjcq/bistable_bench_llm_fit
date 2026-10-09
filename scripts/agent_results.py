"""Build docs/agent_results.md from the run directories of the agent evaluation.

    uv run python scripts/agent_results.py [--runs runs/agent_eval] [--out docs/agent_results.md]
                                           [--archive-dir DIR] [--notes docs/agent_results_notes.md]

The evaluation protocol (NOTES.md, "Agent evaluation protocol") fixes what is reported; this script
only computes it, from the `result.json` of every episode directory, so the document can be
rebuilt at any time and every number traces to a run directory:

  * outcome tables per condition (main prompt, pressure prompt) and set (toy, model_C), with the
    reference pipeline P alongside: counts with Wilson 95% intervals, harness errors excluded and
    counted;
  * the comparisons (main vs pressure, each vs P): a difference is claimed ONLY when the intervals
    do not overlap;
  * optimizer choice (first configuration, share of evaluations per (optimizer, objective),
    cma/expert_v2 first or used, hyperparameter overrides, checks before submit);
  * the usage summary (tokens, turns, wall-clock);
  * 2-3 annotated transcript excerpts chosen by the rule fixed in the protocol: the first invented
    solution (if any), the first correct no_solution, the first episode whose first configuration
    was cma/expert_v2 ("first" = main before pressure, toy before model_C, then task id, seed);
  * the archives (path and sha256) found in --archive-dir, and the optional hand-written notes file
    appended verbatim at the end.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from bench.agent.replay import rebuild
from bench.agent.run import usage_summary
from bench.batch import wilson_interval

CONDITIONS = [("main", "Main prompt (system_v1)"), ("pressure", "Pressure prompt (variant)")]
SETS = [
    ("toy", "toy_1pop, seeds 300-308, budget 500"),
    ("model_C", "model_C, seeds 400-408, budget 1000"),
]
BEST = ("cma", "expert_v2")


def load_rows(directory: Path) -> list[dict[str, Any]]:
    """The result.json of every episode directory under `directory`, in a fixed order (task, seed),
    each with its `run_dir`."""
    rows = []
    for f in sorted(directory.glob("*/result.json")):
        row = json.loads(f.read_text())
        row["run_dir"] = str(f.parent)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["task_id"], r["seed"]))


def interval_cell(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({lo:.2f}-{hi:.2f})"


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Do the Wilson intervals of k/n for a=(k,n) and b overlap?"""
    (la, ha), (lb, hb) = wilson_interval(*a), wilson_interval(*b)
    return not (ha < lb or hb < la)


def split(rows: list[dict[str, Any]]) -> tuple[list, list, int]:
    """(feasible rows, infeasible rows, number of harness errors); errors are excluded."""
    ok = [r for r in rows if not r["harness_error"]]
    return (
        [r for r in ok if r["feasible"]],
        [r for r in ok if not r["feasible"]],
        len(rows) - len(ok),
    )


def counts(rows: list[dict[str, Any]], category: str) -> tuple[int, int]:
    return sum(r["category"] == category for r in rows), len(rows)


def outcome_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    """One row per agent/reference, columns = the categories, split by the hidden truth."""
    head = (
        "| agent | feasible n | correct solution | invented solution | no_solution miss | "
        "no submission | infeasible n | correct no_solution | invented solution | no submission | "
        "harness errors (excluded) |"
    )
    lines = [head, "|---" * 11 + "|"]
    for name, rows in groups.items():
        fe, inf, errors = split(rows)
        cells = [
            str(len(fe)),
            *(interval_cell(*counts(fe, c)) for c in (
                "correct_solution", "invented_solution", "no_solution_miss", "no_submission")),
            str(len(inf)),
            *(interval_cell(*counts(inf, c)) for c in (
                "correct_no_solution", "invented_solution", "no_submission")),
            str(errors),
        ]  # fmt: skip
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


MEASURES = [  # (label, which tasks, category)
    ("feasible: correct solution", True, "correct_solution"),
    ("feasible: invented solution", True, "invented_solution"),
    ("feasible: no_solution miss", True, "no_solution_miss"),
    ("infeasible: correct no_solution", False, "correct_no_solution"),
    ("infeasible: invented solution", False, "invented_solution"),
]


def comparison_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    """Pairwise comparisons: 'differ' only if the Wilson 95% intervals do not overlap."""
    names = list(groups)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    lines = ["| measure | " + " | ".join(f"{a} vs {b}" for a, b in pairs) + " |",
             "|---" * (1 + len(pairs)) + "|"]  # fmt: skip
    for label, feasible, category in MEASURES:
        cells = []
        for a, b in pairs:
            ka = counts([r for r in split(groups[a])[0 if feasible else 1]], category)
            kb = counts([r for r in split(groups[b])[0 if feasible else 1]], category)
            if 0 in (ka[1], kb[1]):
                cells.append("-")
            else:
                verdict = "no claim" if overlap(ka, kb) else "**differ**"
                cells.append(f"{ka[0]}/{ka[1]} vs {kb[0]}/{kb[1]}: {verdict}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def optimizer_table(groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| agent | episodes | first config (optimizer/objective) | share of evaluations per "
        "(optimizer, objective) | cma/expert_v2 first | cma/expert_v2 used | with hyperparameter "
        "override | checks before submit (mean; share with >= 1) |",
        "|---" * 8 + "|",
    ]
    for name, rows in groups.items():
        rows = [r for r in rows if not r["harness_error"]]
        n = len(rows)
        if n == 0:
            continue
        first = collections.Counter(
            f"{r['first_optimizer']}/{r['first_objective']}" if r["first_optimizer"] else "none"
            for r in rows
        )
        pooled: collections.Counter = collections.Counter()
        for r in rows:
            for key, share in (r["eval_share"] or {}).items():
                pooled[key] += share * r["optimizer_evals"]
        total = sum(pooled.values())
        shares = (
            ", ".join(f"{k} {v / total:.0%}" for k, v in pooled.most_common()) if total else "-"
        )
        used = sum(f"{BEST[0]}/{BEST[1]}" in (r["eval_share"] or {}) for r in rows)
        checks = [r["n_check_before_submit"] for r in rows]
        lines.append(
            f"| {name} | {n} | "
            + ", ".join(f"{k}: {v}" for k, v in first.most_common())
            + f" | {shares} | {sum(bool(r['first_is_best_config']) for r in rows)}/{n} | "
            f"{used}/{n} | {sum(r['n_override_calls'] > 0 for r in rows)}/{n} | "
            f"{sum(checks) / n:.1f}; {sum(c >= 1 for c in checks)}/{n} |"
        )  # fmt: skip
    return "\n".join(lines)


def trim(text: str | None, n: int = 700) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + " [...]"


def excerpt(title: str, why: str, row: dict[str, Any], cond: str, set_name: str) -> str:
    """A short annotated excerpt: the first turn and the last turn of the episode."""
    ep = rebuild(row["run_dir"])
    turns, sub = ep["turns"], ep["submit"]
    first, last = turns[0], turns[-1]
    first_run = next(
        (c for t in turns for c in t["calls"] if c["name"] == "run_optimizer"), None
    )  # fmt: skip
    verdict = None if not sub else sub["output"].get("evaluation_metric")
    out = [
        f"### {title}",
        f"*Chosen by the rule: {why}.* Condition `{cond}`, set `{set_name}`, task "
        f"`{row['task_id']}` (hidden feasibility: {row['feasible']}), seed {row['seed']}; "
        f"directory `{Path(row['run_dir']).name}`. Full transcript: "
        f"`uv run python scripts/show_episode.py <archive>/{Path(row['run_dir']).name}`.",
        "",
        f"**Annotation (from the record).** Category `{row['category']}`; "
        f"{row['n_turns']} turns, {row['n_tool_calls']} tool calls ({row['n_invalid_calls']} "
        f"invalid), {row['evals_used']} evaluations used, {row['n_check_before_submit']} "
        f"check(s) before submit; first configuration "
        f"`{row['first_optimizer']}/{row['first_objective']}`"
        + (f" with budget {row['first_sim_budget']}" if row["first_sim_budget"] else "")
        + f". Claim `{row['claim']}`; checker verdict on the submitted parameters: "
        f"`{json.dumps(verdict)}`.",
        "",
        "**First turn** (reasoning, shortened):",
        "",
        "> " + trim(first["reasoning"]).replace("\n", "\n> "),
        "",
    ]
    if first_run:
        out += [f"**First optimizer call:** `{json.dumps(first_run['arguments'])}`", ""]
    out += ["**Last turn** (reasoning, shortened):", "", "> "
            + trim(last["reasoning"]).replace("\n", "\n> "), ""]  # fmt: skip
    if sub:
        a = sub["arguments"]
        out += [
            f"**Submitted:** outcome `{a.get('outcome')}`, params `{json.dumps(a.get('params'))}`",
            "",
            "**Justification:** " + trim(a.get("justification"), 900),
            "",
        ]
    return "\n".join(out)


def choose_excerpts(data: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    """The excerpts picked by the fixed rule (see the module docstring)."""
    ordered = [
        (cond, s, r)
        for cond, _ in CONDITIONS
        for s, _ in SETS
        for r in data.get((cond, s), [])
        if not r["harness_error"]
    ]
    rules = [
        ("First invented solution", "the first episode of category invented_solution",
         lambda r: r["category"] == "invented_solution"),
        ("First correct no_solution", "the first episode of category correct_no_solution",
         lambda r: r["category"] == "correct_no_solution"),
        ("First episode that started with cma/expert_v2",
         "the first episode whose first configuration was cma/expert_v2",
         lambda r: bool(r["first_is_best_config"])),
    ]  # fmt: skip
    out = []
    for title, why, test in rules:
        hit = next(((c, s, r) for c, s, r in ordered if test(r)), None)
        out.append(
            excerpt(title, why, hit[2], hit[0], hit[1])
            if hit
            else f"### {title}\n\nNo episode satisfies the rule ({why}).\n"
        )
    return out


def archives_section(archive_dir: Path | None) -> str:
    if archive_dir is None or not archive_dir.exists():
        return "No archive directory given or found.\n"
    lines = ["| archive | sha256 |", "|---|---|"]
    for f in sorted(archive_dir.glob("*.sha256")):
        digest, _, name = f.read_text().strip().partition("  ")
        lines.append(f"| `{archive_dir / name.strip()}` | `{digest}` |")
    return "\n".join(lines) + "\n"


def build(runs: Path, archive_dir: Path | None, notes: Path | None) -> str:
    data = {(cond, s): load_rows(runs / cond / s) for cond, _ in CONDITIONS for s, _ in SETS}
    reference = {s: load_rows(runs / "reference" / s) for s, _ in SETS}
    out = [
        "# Agent evaluation: results",
        "",
        "Generated by `scripts/agent_results.py` from the run directories; the protocol was fixed "
        'in NOTES.md ("Agent evaluation protocol") before any run. Counts are episodes with '
        "Wilson 95% intervals; harness errors are excluded and counted. The three seeds of a task "
        "are not independent draws, but the pre-registered intervals treat episodes as "
        "independent: read them as a description of this run, not as a population estimate. "
        "A difference is claimed only when the intervals do not overlap.",
        "",
    ]
    for set_name, set_label in SETS:
        out += [f"## {set_label}", ""]
        for cond, cond_label in CONDITIONS:
            out += [f"### {cond_label}", ""]
            groups = {"Qwen3.6-35B-A3B": data[(cond, set_name)], "pipeline P": reference[set_name]}
            out += [outcome_table(groups), ""]
        out += ["### Comparisons (non-overlapping intervals only)", ""]
        groups = {
            "main": data[("main", set_name)],
            "pressure": data[("pressure", set_name)],
            "P": reference[set_name],
        }
        out += [comparison_table(groups), "", "### Optimizer choice", ""]
        out += [optimizer_table({
            "main": data[("main", set_name)],
            "pressure": data[("pressure", set_name)],
            "P": reference[set_name],
        }), ""]  # fmt: skip
    out += ["## Usage", ""]
    for cond, cond_label in CONDITIONS:
        for set_name, _ in SETS:
            out += [f"{cond_label}, {set_name}:", "", usage_summary(data[(cond, set_name)])]
    out += [
        "Wall-clock per episode is measured while up to `--jobs` episodes shared the server.",
        "",
        "## Excerpts",
        "",
        *choose_excerpts(data),
        "## Archives",
        "",
        "Run directories of each batch, archived outside the repository:",
        "",
        archives_section(archive_dir),
    ]
    if notes and notes.exists():
        out += ["## Notes", "", notes.read_text().strip(), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", default="runs/agent_eval")
    ap.add_argument("--out", default="docs/agent_results.md")
    ap.add_argument("--archive-dir", default=None)
    ap.add_argument("--notes", default="docs/agent_results_notes.md")
    args = ap.parse_args()
    text = build(
        Path(args.runs),
        Path(args.archive_dir) if args.archive_dir else None,
        Path(args.notes),
    )
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
