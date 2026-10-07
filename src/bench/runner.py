"""Run one scripted baseline episode (conditions A, B, C) and write its run directory.

    python -m bench.runner CONFIG --task tasks/<file>.json --condition A|B|C --seed S [--out runs/]

Conditions (budget-matched; all spend the whole evaluation budget on ONE optimizer run):
    A  random search, expert objective
    B  optimizer (config baseline.optimizer, or --optimizer), naive objective
    C  same optimizer, expert objective
The best parameters by the condition's own objective are submitted (final_report).

runs/<run_name>/ gets events.jsonl (every call), config.json (everything needed to rebuild the
environment: resolved experiment config, task id, targets, condition, seed) and result.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from bench.config import BaselineConfig, ExperimentConfig, load_config
from bench.env import Environment
from bench.logging import JsonlLogger
from bench.optimizer_runs import RunRegistry
from bench.tasks import Task, load_task

CONDITIONS = ("A", "B", "C")


def condition_setup(condition: str, optimizer: str) -> tuple[str, str]:
    """(optimizer name, objective name) used by a condition."""
    if condition == "A":
        return "random", "expert"
    if condition == "B":
        return optimizer, "naive"
    if condition == "C":
        return optimizer, "expert"
    raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


def run_name(task_id: str, condition: str, optimizer: str, seed: int) -> str:
    """Directory name of a run; condition A has no optimizer choice, so none in its name."""
    label = condition if condition == "A" else f"{condition}-{optimizer}"
    return f"{task_id}__{label}__s{seed}"


def run_baseline(
    cfg: ExperimentConfig,
    task: Task,
    task_id: str,
    condition: str,
    seed: int,
    out_dir: str | Path,
    optimizer: str | None = None,
) -> dict[str, Any]:
    """Run one episode of a baseline condition, write its run directory, return result.json."""
    if optimizer is not None:
        cfg = replace(cfg, baseline=BaselineConfig(optimizer))
    opt_name, objective = condition_setup(condition, cfg.baseline.optimizer)
    run_dir = Path(out_dir) / run_name(task_id, condition, cfg.baseline.optimizer, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": cfg.resolved(),
                "condition": condition,
                "optimizer": opt_name,
                "objective": objective,
                "seed": seed,
                "task": {
                    "id": task_id,
                    "seed": task.seed,
                    "K": task.K,
                    "targets": task.targets.tolist(),  # not the planted params
                },
            },
            indent=2,
        )
    )

    start = time.perf_counter()
    with JsonlLogger(run_dir / "events.jsonl", seed, cfg.hash()) as logger:
        env = Environment.from_task(cfg, task, event_hook=logger.env_hook)
        registry = RunRegistry(env, episode_seed=seed)
        budget = env.budget_remaining
        run = registry.run(opt_name, objective, budget)
        logger.log(
            "harness",
            "run_optimizer",
            {"optimizer": opt_name, "objective": objective, "run_budget": budget},
            {k: v for k, v in run.to_dict().items() if k != "history"},
            env.budget_remaining,
        )
        metric = env.final_report(run.best_params)  # the submission
    result = {
        "run_name": run_dir.name,
        "condition": condition,
        "optimizer": opt_name,
        "objective": objective,
        "seed": seed,
        "task_id": task_id,
        "evals_used": env.n_evaluations,
        "evaluation_metric": asdict(metric),
        "wall_clock_s": time.perf_counter() - start,
        "config_hash": cfg.hash(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", help="experiment YAML, e.g. configs/exp_toy_1pop.yaml")
    ap.add_argument("--task", required=True, help="task file, e.g. tasks/toy_1pop_K2_seed0.json")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="runs", help="directory that receives the run directory")
    ap.add_argument("--optimizer", choices=("cma", "tpe"), help="override baseline.optimizer")
    args = ap.parse_args(argv)
    result = run_baseline(
        load_config(args.config),
        load_task(args.task),
        Path(args.task).stem,
        args.condition,
        args.seed,
        args.out,
        args.optimizer,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
