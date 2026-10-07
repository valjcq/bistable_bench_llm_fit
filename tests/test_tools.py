"""Checkpoint 7: tool layer, schemas, and the scripted dummy agent."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bench.agent.scripted import MARGIN, run_scripted_agent
from bench.config import BudgetConfig
from bench.logging import JsonlLogger, comparable, read_events
from bench.schema import validate
from bench.tasks import load_task
from bench.tools import Episode

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def _episode(cfg, task, budget, path=None, seed=0):
    cfg = replace(cfg, budget=BudgetConfig(budget))
    logger = JsonlLogger(path, seed, cfg.hash()) if path else None
    return Episode(cfg, task, seed, logger)


class Recorder:
    """Wraps Episode.call and records every (tool, arguments) the agent sends."""

    def __init__(self, episode):
        self.episode, self.sent = episode, []
        self.call = episode.call
        episode.call = self._call

    def _call(self, name, arguments=None):
        self.sent.append((name, arguments))
        return self.call(name, arguments)


@pytest.fixture
def toy_task():
    return load_task(TASKS / "toy_1pop_K2_seed4.json")


@pytest.fixture
def ei_task():
    return load_task(TASKS / "ei_2pop_K2_seed1.json")


def test_validator_supported_keywords():
    s = {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "minimum": 1, "maximum": 5},
            "b": {"type": ["string", "null"], "enum": ["x", "y", None]},
            "c": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 2},
            "d": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["a"],
        "additionalProperties": False,
    }
    assert validate({"a": 3, "b": None, "c": [1.5], "d": 0.1}, s) == []
    for bad in (
        {},  # missing a
        {"a": 0}, {"a": 6}, {"a": 1.5}, {"a": True},  # range, type
        {"a": 1, "b": "z"}, {"a": 1, "c": []}, {"a": 1, "c": [1, 2, 3]},
        {"a": 1, "c": ["s"]}, {"a": 1, "d": 0}, {"a": 1, "zzz": 1}, [],
    ):  # fmt: skip
        assert validate(bad, s), bad


def test_schemas_are_plain_json_and_in_the_validator_subset(cfg_toy, toy_task):
    allowed = {
        "type", "properties", "required", "additionalProperties", "enum", "minimum", "maximum",
        "exclusiveMinimum", "items", "minItems", "maxItems", "description",
    }  # fmt: skip

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k not in ("properties",):  # names under "properties" are parameter names
                    yield k
                yield from keys(v if k != "properties" else list(v.values()))
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    specs = _episode(cfg_toy, toy_task, 50).tool_specs()
    json.dumps(specs)
    assert {s["name"] for s in specs} == {
        "get_task", "list_optimizers", "run_optimizer", "get_run_results",
        "check_bistability", "simulate", "get_budget", "submit",
    }  # fmt: skip
    for s in specs:
        assert s["description"] and s["input_schema"]["type"] == "object"
        assert set(keys(s["input_schema"])) <= allowed, s["name"]


@pytest.mark.parametrize(("which", "budget"), [("toy", 60), ("ei", 40)])
def test_dummy_agent_completes_an_episode_with_a_complete_log(
    cfg_toy, cfg_ei, toy_task, ei_task, tmp_path, which, budget
):
    cfg, task = (cfg_toy, toy_task) if which == "toy" else (cfg_ei, ei_task)
    path = tmp_path / "events.jsonl"
    ep = _episode(cfg, task, budget, path)
    rec = Recorder(ep)
    out = run_scripted_agent(ep)
    ep.logger.close()
    assert ep.submitted and out["evaluation_metric"] == ep.final_metric
    events = read_events(path)
    tool_names = [e["name"] for e in events if e["role"] == "agent"]
    assert tool_names[:3] == ["get_task", "list_optimizers", "run_optimizer"]
    assert tool_names[-1] == "submit" and "check_bistability" in tool_names
    assert [e["step"] for e in events] == list(range(len(events)))
    # evaluations: the optimizers' own, plus one per check; none left over except the margin
    n_eval = sum(e["name"] == "evaluate" and e["role"] == "optimizer" for e in events)
    n_check = sum(e["name"] == "check" and e["role"] == "agent" for e in events)
    assert n_eval + n_check == ep.env.n_evaluations
    n_runs = sum(n == "run_optimizer" for n, _ in rec.sent)
    # run 1 = half the budget, +1 check; a second run takes the rest minus the margin
    assert ep.env.n_evaluations == (budget // 2 + 1 if n_runs == 1 else budget - MARGIN)
    left = [e["budget_after"] for e in events]
    assert left == sorted(left, reverse=True)  # the budget only goes down
    assert events[-1]["name"] == "submit" and events[-2]["name"] == "final_report"
    assert all(not e["invalid"] for e in events)
    assert [n for n, _ in rec.sent][-1] == "submit"


def test_dummy_agent_rerun_gives_identical_logs(cfg_toy, toy_task, tmp_path):
    logs = []
    for i in range(2):
        path = tmp_path / f"e{i}.jsonl"
        ep = _episode(cfg_toy, toy_task, 50, path, seed=5)
        run_scripted_agent(ep)
        ep.logger.close()
        logs.append(read_events(path))
    assert comparable(logs[0]) == comparable(logs[1])
    assert logs[0] != logs[1]  # timestamps differ: the comparison is not vacuous


@pytest.mark.parametrize("which", ["toy", "ei"])
def test_every_schema_validates_what_the_dummy_agent_sends(
    cfg_toy, cfg_ei, toy_task, ei_task, which
):
    cfg, task = (cfg_toy, toy_task) if which == "toy" else (cfg_ei, ei_task)
    ep = _episode(cfg, task, 30)
    rec = Recorder(ep)
    run_scripted_agent(ep)
    specs = {s["name"]: s["input_schema"] for s in ep.tool_specs()}
    assert {n for n, _ in rec.sent} >= {"get_task", "run_optimizer", "check_bistability", "submit"}
    for name, args in rec.sent:
        assert validate(args or {}, specs[name]) == [], (name, args)


def test_get_task_never_reveals_the_planted_parameters(cfg_toy, toy_task):
    out = _episode(cfg_toy, toy_task, 50).call("get_task")
    text = json.dumps(out)
    assert "planted" not in text
    for v in toy_task.planted_params.values():
        assert repr(v) not in text
    assert out["budget"] == {"total": 50, "remaining": 50} and out["tol"] == 0.02
    assert [p["name"] for p in out["free_parameters"]] == ["w", "I", "k"]


def test_invalid_calls_are_free_flagged_and_return_errors(cfg_toy, toy_task, tmp_path):
    path = tmp_path / "e.jsonl"
    ep = _episode(cfg_toy, toy_task, 50, path)
    good = {"w": 5.0, "I": 2.0, "k": 1.0}
    bad_calls = [
        ("nope", {}),
        ("check_bistability", {"params": {"w": 5.0}}),  # missing parameters
        ("check_bistability", {"params": {**good, "w": 99.0}}),  # out of bounds
        ("check_bistability", {"params": good, "extra": 1}),
        ("simulate", {"params": good, "init_state": [0.1, 0.2], "T": 10}),  # wrong shape
        ("simulate", {"params": good, "init_state": [0.1], "T": 1e9}),  # above T_max (env check)
        ("run_optimizer", {"name": "nope", "sim_budget": 5, "objective": "expert"}),
        ("run_optimizer", {"name": "cma", "sim_budget": 51, "objective": "expert"}),
        ("run_optimizer", {"name": "cma", "sim_budget": 5, "objective": "success"}),
        ("run_optimizer", {"name": "cma", "sim_budget": 5, "objective": "expert",
                           "hyperparameters": {"popsize": 2}}),
        ("get_run_results", {"run_id": "run9"}),
        ("submit", {"params": {"w": 1.0}}),
        ("get_task", None),  # fine: no arguments is {}
    ]  # fmt: skip
    outs = [ep.call(n, a) for n, a in bad_calls]
    assert all(set(o) == {"error"} and {"type", "message"} == set(o["error"]) for o in outs[:-1])
    assert "error" not in outs[-1]
    assert ep.env.n_evaluations == 0 and not ep.submitted  # nothing was charged
    assert ep.call("get_budget")["remaining"] == 50
    ep.logger.close()
    lines = read_events(path)
    assert [e["invalid"] for e in lines] == [True] * 12 + [False, False]


def test_tools_work_and_get_run_results(cfg_toy, toy_task):
    ep = _episode(cfg_toy, toy_task, 50)
    good = {"w": 10.0, "I": 0.0, "k": 1.0}
    chk = ep.call("check_bistability", {"params": good})
    assert chk["n_stable"] == 2 and len(chk["fixed_points"]) == 3
    sim = ep.call("simulate", {"params": good, "init_state": [0.9], "T": 400})
    assert sim["settled"] and sim["final_state"][0] > 0.99
    assert ep.call("get_budget") == {"total": 50, "used": 2, "remaining": 48}
    assert {o["name"] for o in ep.call("list_optimizers")["optimizers"]} == {"cma", "random", "tpe"}
    run = ep.call("run_optimizer", {"name": "random", "sim_budget": 10, "objective": "naive"})
    assert run["run_id"] == "run1" and run["evals_used"] == 10 and run["budget_remaining"] == 38
    got = ep.call("get_run_results", {"run_id": "run1", "fields": ["history", "seed"]})
    assert set(got) == {"run_id", "history", "seed"} and len(got["history"]) == 10
    assert "history" not in ep.call("get_run_results", {"run_id": "run1"})


def test_budget_exhausted_is_an_error_not_an_exception(cfg_toy, toy_task):
    ep = _episode(cfg_toy, toy_task, 2)
    good = {"w": 10.0, "I": 0.0, "k": 1.0}
    ep.call("check_bistability", {"params": good})
    ep.call("check_bistability", {"params": good})
    out = ep.call("check_bistability", {"params": good})
    assert out["error"]["type"] == "budget_exhausted"
    assert ep.call("get_budget")["remaining"] == 0
    assert "error" not in ep.call("submit", {"params": good})  # submit needs no budget


def test_after_submit_every_tool_returns_an_error(cfg_toy, toy_task, tmp_path):
    path = tmp_path / "e.jsonl"
    ep = _episode(cfg_toy, toy_task, 20, path)
    good = {"w": 10.0, "I": 0.0, "k": 1.0}
    res = ep.call("submit", {"params": good})
    assert set(res["evaluation_metric"]) >= {"success", "target_errors"}
    for spec in ep.tool_specs():
        out = ep.call(spec["name"], {})
        assert out["error"]["type"] == "episode_over", spec["name"]
    assert ep.env.n_evaluations == 0
    ep.logger.close()
    assert all(e["invalid"] for e in read_events(path)[-8:])
