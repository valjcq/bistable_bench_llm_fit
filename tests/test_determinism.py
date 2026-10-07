"""Test 7: same config + seed -> identical task, CheckResult, objective values and hashes.

Everything is rebuilt from the YAML files each time (not reused from fixtures), so this checks
the whole pipeline: loading, planting, evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bench.config import load_config
from bench.env import Environment
from bench.tasks import make_task

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _run(name: str, seed: int):
    cfg = load_config(CONFIGS / f"exp_{name}.yaml")
    task = make_task(cfg, seed=seed)
    env = Environment.from_task(cfg, task)
    params = env.space.from_unit(np.random.default_rng(99).random(env.space.dim))
    n = len(env.describe()["populations"])
    return {
        "hash": env.config_hash,
        "task": task.to_dict(),
        "check": env.check(task.planted_params).to_dict(),
        "check_random": env.check(params).to_dict(),
        "naive": env.evaluate(params, "naive"),
        "expert": env.evaluate(params, "expert"),
        "sim": env.simulate(params, np.full(n, 0.3), 200.0),
    }


@pytest.mark.parametrize("name", ["toy_1pop", "ei_2pop"])
def test_everything_is_reproducible(name):
    a, b = _run(name, 3), _run(name, 3)
    assert a["hash"] == b["hash"]
    assert json.dumps(a["task"]) == json.dumps(b["task"])
    assert json.dumps(a["check"]) == json.dumps(b["check"])
    assert json.dumps(a["check_random"]) == json.dumps(b["check_random"])
    assert a["naive"] == b["naive"] and a["expert"] == b["expert"]  # exact, not approx
    assert a["sim"] == b["sim"]


def test_different_seed_gives_different_task():
    a, b = _run("toy_1pop", 3), _run("toy_1pop", 4)
    assert a["task"]["targets"] != b["task"]["targets"]


def test_evaluation_does_not_depend_on_call_order():
    """The checker's seed lives in the config, so an evaluation never depends on history."""
    cfg = load_config(CONFIGS / "exp_ei_2pop.yaml")
    task = make_task(cfg, seed=0)
    rng = np.random.default_rng(1)
    env1, env2 = Environment.from_task(cfg, task), Environment.from_task(cfg, task)
    p = env1.space.from_unit(rng.random(env1.space.dim))
    q = env1.space.from_unit(rng.random(env1.space.dim))
    first = env1.evaluate(p, "expert")
    env2.evaluate(q, "expert")
    env2.evaluate(q, "naive")
    assert env2.evaluate(p, "expert") == first
