"""Checkpoint 5: the optimizer wrappers and their environment-side adapter."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import bench.optimizers
from bench.budget import BudgetExhausted
from bench.config import BudgetConfig
from bench.env import Environment
from bench.optimizer_runs import RunRegistry, derive_seed
from bench.optimizers import OPTIMIZERS
from bench.optimizers.base import is_converged
from bench.tasks import load_task

TASKS = Path(__file__).resolve().parents[1] / "tasks"
NAMES = sorted(OPTIMIZERS)  # cma, random, tpe


def _toy_env(cfg, budget=200, seed=0):
    """Toy environment with the hand-checked targets of docs section 2: states 0.0 and 1.0."""
    cfg = replace(cfg, budget=BudgetConfig(budget))
    return Environment(cfg, np.array([[0.0], [1.0]]), seed)


def _run(env, name, budget, seed=1, objective="expert_v2", **kw):
    return RunRegistry(env, episode_seed=seed).run(name, objective, budget, seed=seed, **kw)


def test_unit_cube_objective_needs_no_environment():
    """An optimizer works on a bare callable: sphere on the cube, minimum at 0.3."""
    f = lambda x: float(np.sum((x - 0.3) ** 2))  # noqa: E731
    for name, opt in OPTIMIZERS.items():
        r = opt.run(f, d=3, run_budget=120, seed=0)
        assert r.evals_used == 120 and len(r.history) == 120, name
        assert r.best_score == min(r.history)
        assert r.stopped_reason == "run_budget"
    assert OPTIMIZERS["cma"].run(f, 3, 120, 0).best_score < 1e-3


@pytest.mark.parametrize("name", NAMES)
def test_solves_a_committed_toy_task_within_budget(cfg_toy, name):
    """Smoke test on the frozen, in-window toy tasks: >= 1 success in 5 seeds x 5 tasks.

    A single task is not a stable bar: with budget 200 each optimizer solves some tasks and
    fails others (the expert objective is flat in monostable regions), so the bar is "solves at
    least one (task, seed) pair", which still catches a broken wrapper.
    """
    hits = []
    for t in range(5):
        task = load_task(TASKS / f"toy_1pop_K2_seed{t}.json")
        for seed in range(5):
            env = Environment.from_task(cfg_toy, task)
            r = _run(env, name, 200, seed=seed)
            hits.append(env.final_report(r.best_params).success)
    assert any(hits), f"{name} never solved a toy task: {hits}"


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_gives_identical_history(cfg_toy, name):
    runs = [_run(_toy_env(cfg_toy, 60), name, 60, seed=7) for _ in range(2)]
    assert runs[0] == runs[1]
    other = _run(_toy_env(cfg_toy, 60), name, 60, seed=8)
    assert other.history != runs[0].history


@pytest.mark.parametrize("name", NAMES)
def test_init_point_is_evaluated_first(cfg_toy, name):
    env = _toy_env(cfg_toy, 40)
    init = {"w": 7.0, "I": 3.0, "k": 0.5}
    expected = _toy_env(cfg_toy, 40).evaluate(init, "expert_v2")
    r = _run(env, name, 40, init=init)
    assert r.history[0] == expected


def test_cma_without_init_starts_at_the_cube_centre(cfg_toy):
    env = _toy_env(cfg_toy, 20)
    centre = env.space.from_unit([0.5] * env.space.dim)
    first = _toy_env(cfg_toy, 20).evaluate(centre, "expert_v2")
    assert _run(env, "cma", 20).history[0] == first


def test_run_stops_cleanly_when_the_global_budget_runs_out(cfg_toy):
    """Run budget bigger than what is left (the objective raises BudgetExhausted underneath)."""
    env = _toy_env(cfg_toy, 30)

    def f(x):
        return env.evaluate(env.space.from_unit(x), "expert_v2")

    for _ in range(25):  # leave 5 evaluations
        f(np.full(env.space.dim, 0.5))
    r = OPTIMIZERS["tpe"].run(f, env.space.dim, run_budget=50, seed=0)
    assert r.stopped_reason == "global_budget"
    assert r.evals_used == 5 and r.best_score is not None
    with pytest.raises(BudgetExhausted):
        f(np.full(env.space.dim, 0.5))


def test_run_budget_above_remaining_is_a_free_value_error(cfg_toy):
    env = _toy_env(cfg_toy, 30)
    reg = RunRegistry(env, 0)
    with pytest.raises(ValueError, match="remaining"):
        reg.run("random", "expert_v2", 31)
    assert env.n_evaluations == 0
    for bad in (
        dict(optimizer="nope", objective="expert_v2", run_budget=5),
        dict(optimizer="random", objective="success", run_budget=5),
        dict(optimizer="random", objective="expert_v2", run_budget=0),
        dict(optimizer="random", objective="expert_v2", run_budget=5, hyperparameters={"x": 1}),
        dict(optimizer="cma", objective="expert_v2", run_budget=5, hyperparameters={"sigma0": 9}),
        dict(optimizer="random", objective="expert_v2", run_budget=5, init={"w": 1.0}),
    ):
        with pytest.raises(ValueError):
            reg.run(**bad)
    assert env.n_evaluations == 0


def test_converged_flag_definition():
    flat = [1.0] * 50
    assert is_converged(flat, 100)  # window max(20, 20) = 20, no improvement
    assert not is_converged([1.0] * 20, 100)  # not enough evaluations to fill the window
    improving = [1.0 - 0.01 * i for i in range(50)]
    assert not is_converged(improving, 100)
    assert is_converged([1.0] * 30 + [1.0 - 1e-7] * 30, 100)  # below 1e-6: still stagnant


def test_stop_on_converged_ends_the_run_early():
    r = OPTIMIZERS["random"].run(lambda x: 1.0, 2, 100, 0, stop_on_converged=True)
    assert r.stopped_reason == "converged" and r.evals_used == 21 and r.converged


def test_registry_seeds_and_results(cfg_toy):
    env = _toy_env(cfg_toy, 60)
    reg = RunRegistry(env, episode_seed=3)
    a = reg.run("random", "naive", 10)
    b = reg.run("random", "naive", 10)
    assert (a.run_id, b.run_id) == ("run1", "run2")
    assert a.seed == derive_seed(3, 1) != b.seed
    assert reg.get("run1") is a
    assert set(a.best_params) == {"w", "I", "k"}
    with pytest.raises(ValueError):
        reg.get("run9")
    assert env.n_evaluations == 20


def test_random_lhs_and_hyperparameter_defaults(cfg_toy):
    r = _run(_toy_env(cfg_toy, 30), "random", 30, hyperparameters={"sampling": "lhs"})
    xs = np.array(r.history)
    assert r.hyperparameters == {"sampling": "lhs"} and len(xs) == 30
    assert _run(_toy_env(cfg_toy, 5), "cma", 5).hyperparameters == {"sigma0": 0.25, "popsize": None}


def test_cma_runs_at_the_minimum_popsize(cfg_toy):
    r = _run(_toy_env(cfg_toy, 40), "cma", 40, hyperparameters={"popsize": 4})
    assert r.evals_used == 40 and r.hyperparameters["popsize"] == 4
    for bad in (2, 3):
        with pytest.raises(ValueError, match="popsize"):
            _run(_toy_env(cfg_toy, 40), "cma", 40, hyperparameters={"popsize": bad})


def test_specs_are_json_serializable():
    import json

    for opt in OPTIMIZERS.values():
        d = opt.spec.to_dict()
        json.dumps(d)
        assert d["name"] == opt.spec.name and d["hyperparameters"]


def test_optimizers_never_import_the_environment_side():
    """Static check: no module of bench.optimizers imports model, checker, env, tasks, ..."""
    forbidden = {"env", "checker", "models", "tasks", "scoring", "simulate", "config"}
    for path in Path(bench.optimizers.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                assert not (parts[0] == "bench" and len(parts) > 1 and parts[1] in forbidden), path
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    assert not (parts[0] == "bench" and len(parts) > 1 and parts[1] in forbidden)


def test_optimizers_never_see_planted_params(cfg_toy, task_toy):
    """The environment is built from targets + seed only; the objective is the sole channel."""
    env = Environment.from_task(cfg_toy, task_toy)
    r = RunRegistry(env, 0).run("random", "expert_v2", 20)
    planted = np.array(list(task_toy.planted_params.values()))
    assert not any(np.allclose(list(p.values()), planted) for p in [r.best_params])
    assert not hasattr(env, "planted_params") and not hasattr(env, "_task")
