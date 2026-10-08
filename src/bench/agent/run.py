"""Run LLM-agent episodes: one task, or a batch over a task set, with a real backend or --dry-run.

    python -m bench.agent.run CONFIG --task FILE --seed S [--agent AGENT.yaml] [--dry-run]
    python -m bench.agent.run CONFIG --tasks 'tasks/honesty/toy*.json' --seeds 0-2 --jobs 4 ...

CONFIG is an experiment YAML (model, task, budget, ...) that may contain an `agent:` section; a
separate `--agent` YAML (either a mapping with an `agent:` key or the section itself) supplies or
replaces it. The agent section is part of the resolved config, hence of the config hash, and of each
run's config.json. `--dry-run` swaps the backend for the scripted FakeBackend (no network, no key).

Every episode is a run directory (events.jsonl with the complete transcript, config.json,
result.json), a CSV row, and the batch ends with a usage summary (turns, tokens, wall-clock).
The honesty metrics come from bench.honesty. LLM runs are not bit-reproducible: the logged
transcripts and configs are the record (docs/agent.md).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from bench.agent.backends import make_backend
from bench.agent.loop import load_system_prompt, run_agent
from bench.batch import parse_seeds
from bench.config import AgentConfig, BudgetConfig, ExperimentConfig, load_config
from bench.honesty import CSV_FIELDS, run_honesty_episode, summarize
from bench.tasks import load_task


def resolve_agent(cfg: ExperimentConfig, agent_file: str | None, dry_run: bool) -> ExperimentConfig:
    """The experiment config with its final `agent` section (file override, dry-run fake)."""
    agent = cfg.agent
    if agent_file:
        raw = yaml.safe_load(Path(agent_file).read_text())
        agent = AgentConfig(**(raw.get("agent", raw)))
    if dry_run:
        agent = replace(
            agent or AgentConfig("fake", "fake"),
            backend="fake", model="fake", base_url=None, api_key_env=None,
        )  # fmt: skip
    if agent is None:
        raise SystemExit("no agent section: put one in CONFIG, pass --agent FILE, or use --dry-run")
    return replace(cfg, agent=agent)


def run_label(agent: AgentConfig) -> str:
    """Short name for run directories: dryrun, or the model name made filesystem-safe."""
    return "dryrun" if agent.backend == "fake" else re.sub(r"[^A-Za-z0-9._-]+", "-", agent.model)


def run_one(
    cfg: ExperimentConfig, task_path: str | Path, seed: int, out_dir: str | Path
) -> dict[str, Any]:
    """One agent episode on one task; returns the CSV row (the run directory is written)."""
    agent = cfg.agent
    assert agent is not None
    task, task_id = load_task(task_path), Path(task_path).stem
    prompt = load_system_prompt(agent.prompt)
    backend = make_backend(agent)

    def play(episode: Any) -> Any:
        return run_agent(episode, backend, prompt, max_turns=agent.max_turns)

    return run_honesty_episode(
        cfg, task, task_id, seed, out_dir, run_label(agent), agent.max_tool_calls, play,
        extra_config={
            "agent_label": run_label(agent),
            "prompt_file": f"prompts/{agent.prompt}.md",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        },
    )  # fmt: skip


def usage_summary(rows: list[dict[str, Any]]) -> str:
    """Markdown cost/usage summary of a batch: episodes, turns, tokens, model time, wall-clock."""
    n = len(rows)
    if n == 0:
        return "no episodes\n"

    def tot(key: str) -> float:
        return sum(float(r[key]) for r in rows)

    errors = sum(bool(r["harness_error"]) for r in rows)
    lines = [
        "| episodes | harness errors | turns (total / mean) | tokens in (total / mean) | "
        "tokens out (total / mean) | model time s (total) | retries | "
        "wall-clock s (total / mean) |",
        "|---|---|---|---|---|---|---|---|",
        f"| {n} | {errors} | {tot('n_turns'):.0f} / {tot('n_turns') / n:.1f} | "
        f"{tot('tokens_in'):.0f} / {tot('tokens_in') / n:.0f} | "
        f"{tot('tokens_out'):.0f} / {tot('tokens_out') / n:.0f} | {tot('model_latency_s'):.1f} | "
        f"{tot('retries'):.0f} | {tot('wall_clock_s'):.1f} / {tot('wall_clock_s') / n:.1f} |",
    ]
    return "\n".join(lines) + "\n"


def _job(args: tuple) -> dict[str, Any]:
    return run_one(*args)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", help="experiment YAML (may contain an agent: section)")
    ap.add_argument("--agent", help="agent YAML supplying or replacing the agent section")
    ap.add_argument("--task", help="one task file")
    ap.add_argument("--seed", type=int, help="seed of that episode")
    ap.add_argument("--tasks", nargs="+", help="batch: task files (globs allowed)")
    ap.add_argument("--seeds", help="batch: e.g. 0-2")
    ap.add_argument("--budget", type=int, help="override budget.max_evaluations")
    ap.add_argument("--dry-run", action="store_true", help="use the scripted FakeBackend")
    ap.add_argument("--jobs", type=int, default=1, help="parallel episodes (batch)")
    ap.add_argument("--out", default="runs/agent")
    ap.add_argument("--csv", default=None, help="results CSV (default: <out>/results.csv)")
    args = ap.parse_args(argv)
    if bool(args.task) == bool(args.tasks):
        ap.error("give either --task FILE --seed S, or --tasks GLOB... --seeds A-B")
    cfg = resolve_agent(load_config(args.config), args.agent, args.dry_run)
    if args.budget is not None:
        cfg = replace(cfg, budget=BudgetConfig(args.budget))
    if args.task:
        if args.seed is None:
            ap.error("--task needs --seed")
        work = [(cfg, args.task, args.seed, args.out)]
    else:
        if not args.seeds:
            ap.error("--tasks needs --seeds")
        paths = sorted(p for pat in args.tasks for p in glob.glob(pat))
        work = [(cfg, p, s, args.out) for p in paths for s in parse_seeds(args.seeds)]

    if args.jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_job, work))
    else:
        rows = [_job(w) for w in work]

    csv_path = Path(args.csv or Path(args.out) / "results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: json.dumps(r[k]) if isinstance(r[k], dict | list) else r[k] for k in CSV_FIELDS}
            )
    usage = usage_summary(rows)
    csv_path.with_name(csv_path.stem + "_usage.md").write_text(usage)
    print(summarize(rows))
    print()
    print(usage)
    print(f"agent: {asdict(cfg.agent)}\nconfig hash: {cfg.hash()}")


if __name__ == "__main__":
    main()
