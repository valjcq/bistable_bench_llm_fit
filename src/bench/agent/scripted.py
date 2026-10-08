"""The scripted dummy agent: a fixed sequence of tool calls, no LLM.

It exists to prove the whole pipeline (tools, logger, episode end) before any model is
involved, and it goes through exactly the same `Episode.call` an LLM agent will use. Its policy:

  get_task -> list_optimizers -> run_optimizer(cma, expert, half the budget)
  -> check_bistability(best) -> if that did not succeed:
       run_optimizer(tpe, expert, init=best, rest of the budget minus a margin)
  -> submit(best)

The second run starts from the first run's best point, so it can only match or improve it.
"""

from __future__ import annotations

from typing import Any

from bench.tools import Episode

MARGIN = 5  # evaluations deliberately left unspent after the second run


def run_scripted_agent(episode: Episode) -> dict[str, Any]:
    """Play one full episode on `episode`; returns the result of submit."""
    call = episode.call
    task = call("get_task")
    call("list_optimizers")
    first = call(
        "run_optimizer",
        {
            "name": "cma",
            "hyperparameters": {},
            "sim_budget": task["budget"]["total"] // 2,
            "objective": "expert_v2",
        },
    )
    best = first["best_params"]
    if not call("check_bistability", {"params": best})["success"]:
        rest = call("get_budget")["remaining"] - MARGIN
        if rest >= 1:
            second = call(
                "run_optimizer",
                {"name": "tpe", "sim_budget": rest, "objective": "expert_v2", "init": best},
            )
            best = second["best_params"]
    return call("submit", {"outcome": "solution", "params": best})
