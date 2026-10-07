"""Test 6: the budget counts every call, raises at the limit, and cannot be bypassed."""

from dataclasses import replace

import numpy as np
import pytest

from bench.budget import Budget, BudgetExhausted
from bench.config import BudgetConfig
from bench.env import Environment
from bench.models.rate_network import RateNetwork
from bench.tasks import Task


def _env(cfg, task, budget=50, hook=None):
    return Environment.from_task(replace(cfg, budget=BudgetConfig(budget)), task, event_hook=hook)


def _calls(env, params):
    """One call of each budgeted kind."""
    n = env._model.n_pops  # test-only peek
    return {
        "evaluate-naive": lambda: env.evaluate(params, "naive"),
        "evaluate-expert": lambda: env.evaluate(params, "expert"),
        "check": lambda: env.check(params),
        "simulate": lambda: env.simulate(params, np.full(n, 0.5), 50.0),
    }


def test_every_budgeted_call_costs_exactly_one(cfg_toy, task_toy):
    env = _env(cfg_toy, task_toy)
    for i, call in enumerate(_calls(env, task_toy.planted_params).values(), start=1):
        call()
        assert env.n_evaluations == i
    assert env.budget_remaining == 50 - 4
    # naive simulates K trajectories but is still ONE evaluation (checked by the loop above).


def test_invalid_calls_and_final_report_are_free(cfg_toy, task_toy):
    env = _env(cfg_toy, task_toy)
    good = task_toy.planted_params
    with pytest.raises(ValueError):
        env.evaluate(good, "success")  # unknown objective: the final metric is not offered
    with pytest.raises(ValueError):
        env.evaluate({**good, "w": 1e6}, "expert")  # out of bounds
    with pytest.raises(ValueError):
        env.evaluate({"w": 1.0}, "expert")  # missing parameters
    with pytest.raises(ValueError):
        env.simulate(good, np.zeros(5), 10.0)  # wrong x0 shape
    env.final_report(good)
    assert env.n_evaluations == 0


def test_raises_at_the_limit_for_every_method_without_doing_work(cfg_ei, task_ei):
    events = []
    env = _env(cfg_ei, task_ei, budget=3, hook=events.append)
    params = task_ei.planted_params
    calls = _calls(env, params)
    for _ in range(3):
        env.evaluate(params, "expert")
    assert env.budget_remaining == 0
    before = len(events)
    for name, call in calls.items():
        with pytest.raises(BudgetExhausted):
            call()
        assert env.n_evaluations == 3, name  # nothing was charged or computed
    refused = events[before:]
    assert [e["event"] for e in refused] == ["budget_exhausted"] * 4
    assert [e["attempted"] for e in refused] == ["evaluate", "evaluate", "check", "simulate"]
    env.final_report(params)  # reporting still works after the budget is gone


def test_budget_object_cannot_be_refunded():
    b = Budget(2)
    b.charge()
    b.charge()
    with pytest.raises(BudgetExhausted):
        b.charge()
    assert b.used == 2 and b.remaining == 0
    with pytest.raises(AttributeError):
        b.used = 0  # read-only property


def test_public_api_does_not_expose_model_budget_or_planted_params(cfg_toy, task_toy):
    env = _env(cfg_toy, task_toy)
    public = [n for n in dir(env) if not n.startswith("_")]
    for name in public:
        value = getattr(env, name)
        assert not isinstance(value, (RateNetwork, Budget, Task)), name
    for forbidden in ("budget", "model", "task", "planted_params", "net"):
        assert not hasattr(env, forbidden)
    # Nothing the environment returns carries the hidden parameters.
    planted_values = {repr(v) for v in task_toy.planted_params.values()}
    exposed = repr(env.describe()) + repr(env.targets) + repr(env.space)
    assert not any(v in exposed for v in planted_values)
    # Tasks are only used for their targets: the Task object is not kept.
    assert not any(isinstance(v, Task) for v in vars(env).values())


def test_returned_objects_cannot_be_used_to_alter_the_environment(cfg_toy, task_toy):
    env = _env(cfg_toy, task_toy)
    t = env.targets
    t[:] = 123.0
    assert not np.allclose(env.targets, 123.0)  # targets is a copy
    with pytest.raises(TypeError):
        env.space.fixed["tau"] = 1.0  # type: ignore[index]  # read-only mapping
    assert env.describe() is not env.describe()  # fresh dict each time


def test_event_hook_sees_every_call(cfg_toy, task_toy):
    events = []
    env = _env(cfg_toy, task_toy, hook=events.append)
    p = task_toy.planted_params
    env.evaluate(p, "expert")
    env.check(p)
    env.simulate(p, np.array([0.5]), 30.0)
    env.final_report(p)
    assert [e["event"] for e in events] == ["evaluate", "check", "simulate", "final_report"]
    assert [e["index"] for e in events] == [1, 2, 3, None]
    assert events[0]["params"] == pytest.approx(p)


def test_task_from_other_settings_is_refused(cfg_toy, cfg_ei, task_ei):
    with pytest.raises(ValueError, match="different model"):
        Environment.from_task(cfg_toy, task_ei)


def test_final_report_is_not_reachable_through_the_budgeted_interface(cfg_toy, task_toy):
    """final_report is unbudgeted, so no budgeted method may lead to it (that would give an
    optimizer a free, flat success oracle)."""
    import inspect

    from bench.env import BUDGETED_METHODS, OBJECTIVES

    assert "final_report" not in BUDGETED_METHODS
    assert set(BUDGETED_METHODS) == {"evaluate", "check", "simulate"}
    # The objective names cannot be used to ask for the final metric.
    assert not {"success", "final", "final_report", "metric", "evaluation_metric"} & set(OBJECTIVES)
    # No budgeted method has a parameter that could select it.
    for name in BUDGETED_METHODS:
        sig = inspect.signature(getattr(Environment, name))
        assert set(sig.parameters) <= {"self", "params", "objective_name", "x0", "T"}, name

    # Behaviourally: running every budgeted call never triggers a final_report event, while
    # calling it directly does (and does not touch the budget).
    events = []
    env = _env(cfg_toy, task_toy, hook=events.append)
    p = task_toy.planted_params
    for call in _calls(env, p).values():
        call()
    assert "final_report" not in [e["event"] for e in events]
    for bad in ("success", "final_report", "evaluation_metric"):
        with pytest.raises(ValueError):
            env.evaluate(p, bad)
    used = env.n_evaluations
    env.final_report(p)
    assert env.n_evaluations == used
    assert events[-1]["event"] == "final_report"
