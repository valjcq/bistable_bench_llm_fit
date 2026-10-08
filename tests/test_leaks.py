"""Leak check: nothing an agent can see reveals the planting window, the reference range, the
feasibility flag or how tasks were generated.

Everything the tools can return is collected for several tasks (feasible and infeasible, toy and
wc1972): get_task, every other tool's output, the error messages of failing calls, the tool specs
(names, descriptions, schemas) and the optimizer specs. `get_task` must have exactly the expected
keys, and no collected text may contain a forbidden word.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from bench.config import BudgetConfig, load_config
from bench.optimizers import OPTIMIZERS
from bench.tasks import load_task
from bench.tools import Episode

ROOT = Path(__file__).resolve().parents[1]
HONESTY = ROOT / "tasks/honesty"
GET_TASK_KEYS = {
    "model", "populations", "types", "observed", "K", "tol", "targets", "free_parameters",
    "budget", "max_tool_calls",
}  # fmt: skip
FORBIDDEN = (
    "window", "reference", "feasib", "plant", "generated", "proof", "hidden", "tol_fraction",
    "min_abs", "task_inputs", "lowest_target", "state bound", "state_bound", "secant", "logit",
    # no paper and no known model may be named (the internal names included)
    "wilson", "cowan", "wc1972", "toy_1pop", "ei_2pop", "wc_shifted", "model card",
)  # fmt: skip
CASES = [
    ("exp_toy_1pop.yaml", "toy_1pop_K2_seed300.json"),  # infeasible
    ("exp_toy_1pop.yaml", "toy_1pop_K2_seed302.json"),  # feasible
    ("exp_wc1972.yaml", "wc1972_K2_seed400.json"),  # infeasible
    ("exp_wc1972.yaml", "wc1972_K2_seed401.json"),  # feasible
]


def everything_an_agent_sees(config: str, task_file: str) -> tuple[dict, str]:
    cfg = replace(load_config(ROOT / "configs" / config), budget=BudgetConfig(12))
    ep = Episode(cfg, load_task(HONESTY / task_file), 0, max_tool_calls=30)
    names = [p.name for p in ep.env.space.free]
    good = {n: float(sum(ep.env.space.free[i].low + ep.env.space.free[i].high for i in [j]) / 2)
            for j, n in enumerate(names)}  # fmt: skip
    task = ep.call("get_task")
    n_pops = len(task["populations"])
    outputs = [
        task,
        ep.call("list_optimizers"),
        ep.call("get_budget"),
        ep.call("check_bistability", {"params": good}),
        ep.call("simulate", {"params": good, "init_state": [0.1] * n_pops, "T": 100}),
        ep.call("run_optimizer", {"name": "random", "sim_budget": 3, "objective": "expert_v2"}),
        ep.call("run_optimizer", {"name": "cma", "sim_budget": 3, "objective": "naive"}),
        ep.call("get_run_results", {"run_id": "run1", "fields": ["history", "best_params"]}),
        # failing calls: their messages are agent-visible too
        ep.call("check_bistability", {"params": {**good, names[0]: 1e9}}),
        ep.call("check_bistability", {"params": {names[0]: 1.0}}),
        ep.call("simulate", {"params": good, "init_state": [0.1] * n_pops, "T": 1e9}),
        ep.call(
            "simulate",
            {"params": good, "init_state": [0.1], "T": 5}
            if n_pops > 1
            else {"params": good, "init_state": [0.1, 0.2], "T": 5},
        ),  # fmt: skip
        ep.call("run_optimizer", {"name": "nope", "sim_budget": 3, "objective": "naive"}),
        ep.call("run_optimizer", {"name": "cma", "sim_budget": 3, "objective": "success"}),
        ep.call("run_optimizer", {"name": "cma", "sim_budget": 999, "objective": "naive"}),
        ep.call(
            "run_optimizer",
            {
                "name": "cma",
                "sim_budget": 2,
                "objective": "naive",
                "hyperparameters": {"popsize": 1},
            },
        ),  # fmt: skip
        ep.call("get_run_results", {"run_id": "run99"}),
        ep.call("nope"),
        ep.call("submit", {"outcome": "solution"}),
        ep.call("submit", {"outcome": "no_solution", "justification": "x"}),
        ep.call("get_task"),  # after submit: episode_over
    ]
    texts = outputs + [
        ep.tool_specs(),
        [o.spec.to_dict() for o in OPTIMIZERS.values()],
    ]  # fmt: skip
    return task, json.dumps(texts).lower()


@pytest.mark.parametrize(("config", "task_file"), CASES)
def test_get_task_has_exactly_the_expected_keys(config, task_file):
    task, _ = everything_an_agent_sees(config, task_file)
    expected = GET_TASK_KEYS | ({"inputs"} if "wc1972" in task_file else set())
    assert set(task) == expected


@pytest.mark.parametrize(("config", "task_file"), CASES)
def test_nothing_an_agent_sees_contains_a_forbidden_word(config, task_file):
    _, text = everything_an_agent_sees(config, task_file)
    for word in FORBIDDEN:
        assert word not in text, word
    # the year, as a whole token (not inside a number such as 0.31972)
    assert not re.search(r"(?<![0-9.])1972(?![0-9])", text)


@pytest.mark.parametrize(
    ("config", "task_file", "label"),
    [
        ("exp_toy_1pop.yaml", "toy_1pop_K2_seed302.json", "model_A"),
        ("exp_wc1972.yaml", "wc1972_K2_seed401.json", "model_C"),
    ],
)
def test_get_task_shows_a_neutral_model_label(config, task_file, label):
    task, _ = everything_an_agent_sees(config, task_file)
    assert task["model"] == label


def test_every_internal_model_has_a_public_label():
    from bench.models.instances import public_label

    assert [public_label(n) for n in ("toy_1pop", "ei_2pop", "wc1972", "wc1972_k1")] == [
        "model_A", "model_B", "model_C", "model_C",
    ]  # fmt: skip
    with pytest.raises(ValueError):
        public_label("something_else")


def test_the_leak_check_would_catch_a_leak():
    """Sanity: the collector does see error messages and get_task, so a leak would show up."""
    _, text = everything_an_agent_sees("exp_toy_1pop.yaml", "toy_1pop_K2_seed300.json")
    assert "unknown tool" in text and "must be <=" in text and "max_tool_calls" in text
