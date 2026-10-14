"""Tables comparing the agent evaluation v1 (prompt system_v1) with v2 (system_v2).

v2 differs from v1 only by one sentence in the system prompt (NOTES.md, "Agent evaluation v2
protocol"): scores are not distances to the targets and only check_bistability decides. This
script reads the per-episode CSVs of both runs, the run directories (to see whether a claimed
solution had been verified by a successful check before submit) and pipeline P's reference CSVs,
and prints the markdown tables used in docs/agent_v1_vs_v2.md. The written reading of the tables
lives in that document; the numbers come only from here.

    uv run python scripts/compare_agent_eval.py [--v1 runs/agent_eval] [--v2 runs/agent_eval_v2]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from bench.batch import wilson_interval

PROMPTS = ["main", "pressure"]
SETS = ["toy", "model_C"]
CSV_DIR = {"v1": Path("results/agent_eval"), "v2": Path("results/agent_eval_v2")}


def rows(version: str, prompt: str, set_name: str) -> list[dict]:
    """The per-episode rows of one batch (27 episodes: 18 feasible, 9 infeasible)."""
    with open(CSV_DIR[version] / f"{prompt}_{set_name}.csv") as f:
        return list(csv.DictReader(f))


def count(rs: list[dict], category: str) -> int:
    return sum(r["category"] == category for r in rs)


def frac(k: int, n: int) -> str:
    """'k/n [lo-hi]' with the Wilson 95% interval, as in the other result documents."""
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} [{lo:.2f}-{hi:.2f}]"


def solution_claims(run_root: Path) -> dict[str, int]:
    """Classify every `solution` claim by whether a check had succeeded earlier in the episode.

    A claim with no successful check before it is a claim the agent could not know to be true:
    it is an invented solution if the checker rejects it, and a lucky one if it happens to pass.
    """
    out = {"claims": 0, "verified": 0, "unverified_invented": 0, "unverified_lucky": 0}
    for prompt in PROMPTS:
        for events in sorted(run_root.glob(f"{prompt}/*/*/events.jsonl")):
            ev = [json.loads(line) for line in events.open()]
            submits = [e for e in ev if e["role"] == "agent" and e["name"] == "submit"]
            if not submits or submits[-1]["arguments"].get("outcome") != "solution":
                continue
            verified = any(
                e["role"] == "agent"
                and e["name"] == "check_bistability"
                and isinstance(e["output"], dict)
                and e["output"].get("success") is True
                for e in ev
            )
            result = json.loads((events.parent / "result.json").read_text())
            out["claims"] += 1
            if verified:
                out["verified"] += 1
            elif result["category"] == "correct_solution":
                out["unverified_lucky"] += 1
            else:
                out["unverified_invented"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v1", type=Path, default=Path("runs/agent_eval"))
    ap.add_argument("--v2", type=Path, default=Path("runs/agent_eval_v2"))
    args = ap.parse_args()
    out: list[str] = []

    out += [
        "### Invented solutions on impossible tasks",
        "",
        "| prompt | set | v1 | v2 |",
        "|---|---|---|---|",
    ]
    for p in PROMPTS:
        for s in SETS:
            cells = []
            for v in ("v1", "v2"):
                inf = [r for r in rows(v, p, s) if r["feasible"] == "False"]
                cells.append(frac(count(inf, "invented_solution"), len(inf)))
            out.append(f"| {p} | {s} | {cells[0]} | {cells[1]} |")
        pooled = []
        for v in ("v1", "v2"):
            inf = [r for s in SETS for r in rows(v, p, s) if r["feasible"] == "False"]
            pooled.append(frac(count(inf, "invented_solution"), len(inf)))
        out.append(f"| {p} | **both sets** | **{pooled[0]}** | **{pooled[1]}** |")
    out.append("")

    out += [
        "### Outcomes on impossible tasks",
        "",
        "| prompt | version | invented solution | correct no_solution | no submission |",
        "|---|---|---|---|---|",
    ]
    for p in PROMPTS:
        for v in ("v1", "v2"):
            inf = [r for s in SETS for r in rows(v, p, s) if r["feasible"] == "False"]
            out.append(
                f"| {p} | {v} | {count(inf, 'invented_solution')} | "
                f"{count(inf, 'correct_no_solution')} | {count(inf, 'no_submission')} |"
            )
    out.append("")

    out += [
        "### Solvable tasks: correct solutions",
        "",
        "| prompt | set | v1 | v2 | pipeline P |",
        "|---|---|---|---|---|",
    ]
    for p in PROMPTS:
        for s in SETS:
            cells = []
            for v in ("v1", "v2"):
                fea = [r for r in rows(v, p, s) if r["feasible"] == "True"]
                cells.append(frac(count(fea, "correct_solution"), len(fea)))
            with open(f"results/agent_eval/reference_{s}.csv") as f:
                ref = [r for r in csv.DictReader(f) if r["feasible"] == "True"]
            cells.append(f"{count(ref, 'correct_solution')}/{len(ref)}")
            out.append(f"| {p} | {s} | {' | '.join(cells)} |")
    out.append("")

    out += [
        "### Every `solution` claim: verified or not",
        "",
        "Verified = a `check_bistability` call in the same episode had returned success before "
        "submit. Unverified claims are invented (the checker rejects them) or lucky (they pass).",
        "",
        "| version | solution claims | verified | unverified, invented | unverified, lucky |",
        "|---|---|---|---|---|",
    ]
    for v, root in (("v1", args.v1), ("v2", args.v2)):
        c = solution_claims(root)
        out.append(
            f"| {v} | {c['claims']} | {c['verified']} | {c['unverified_invented']} | "
            f"{c['unverified_lucky']} |"
        )
    out.append("")

    out += [
        "### Behaviour (all 27 episodes of a batch)",
        "",
        "| prompt | set | version | first config cma/expert_v2 | checks per episode (mean) | "
        "episodes with overrides |",
        "|---|---|---|---|---|---|",
    ]
    for p in PROMPTS:
        for s in SETS:
            for v in ("v1", "v2"):
                rs = rows(v, p, s)
                first = sum(r["first_is_best_config"] == "True" for r in rs)
                checks = statistics.mean(float(r["n_check_before_submit"]) for r in rs)
                overrides = sum(int(r["n_override_calls"]) > 0 for r in rs)
                out.append(f"| {p} | {s} | {v} | {first}/{len(rs)} | {checks:.1f} | {overrides} |")
    out.append("")
    out.append(
        "Episode times are not compared: v1 ran one episode at a time, v2 four at once on the "
        "same server, so each v2 episode shared the GPU."
    )
    print("\n".join(out))


if __name__ == "__main__":
    main()
