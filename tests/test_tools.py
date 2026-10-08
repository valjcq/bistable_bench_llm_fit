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
        "exclusiveMinimum", "items", "minItems", "maxItems", "description", "$schema", "maxLength",
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
    # group by call: each tool call writes one line with its call_id; the environment events
    # it triggered carry that id as parent_call_id and come before it
    calls = [e for e in events if e["call_id"] is not None]
    assert [c["call_id"] for c in calls] == list(range(1, len(calls) + 1))
    children = {c["call_id"]: [e for e in events if e["parent_call_id"] == c["call_id"]]
                for c in calls}  # fmt: skip
    for c in calls:
        assert all(e["step"] < c["step"] for e in children[c["call_id"]])
        kinds = [e["name"] for e in children[c["call_id"]]]
        if c["name"] == "run_optimizer":
            want = ["evaluate"] * c["output"]["evals_used"]
        else:
            want = {"check_bistability": ["check"], "submit": ["final_report"]}.get(c["name"], [])
        assert kinds == want, c["name"]
    assert all(e["parent_call_id"] is None for e in calls)
    assert sum(len(v) for v in children.values()) + len(calls) == len(events)  # nothing orphaned
    n_charged = sum(e["name"] in ("evaluate", "check") for e in events)
    assert n_charged == ep.env.n_evaluations
    n_runs = sum(n == "run_optimizer" for n, _ in rec.sent)
    # run 1 = half the budget, +1 check; a second run takes the rest minus the margin
    assert ep.env.n_evaluations == (budget // 2 + 1 if n_runs == 1 else budget - MARGIN)
    left = [e["budget_after"] for e in events]
    assert left == sorted(left, reverse=True)  # the budget only goes down
    assert events[-1]["name"] == "submit" and events[-2]["name"] == "final_report"
    assert all(not e["invalid"] for e in events)
    assert [n for n, _ in rec.sent][-1] == "submit"


def test_schemas_pass_the_official_metaschema_and_agree_with_the_runtime_validator(
    cfg_toy, toy_task
):
    from jsonschema import Draft202012Validator

    from bench.tools import JSON_SCHEMA_DRAFT

    good = {"w": 5.0, "I": 2.0, "k": 1.0}
    cases = {  # tool -> (valid examples, invalid examples)
        "get_task": ([{}], [{"x": 1}, [], "s", {"params": good}]),
        "list_optimizers": ([{}], [{"x": 1}, None]),
        "get_budget": ([{}], [{"verbose": True}]),
        "run_optimizer": (
            [
                {"name": "cma", "sim_budget": 10, "objective": "expert_v2"},
                {"name": "tpe", "sim_budget": 50, "objective": "naive", "init": good,
                 "hyperparameters": {"multivariate": False}, "seed": 3},
                {"name": "random", "sim_budget": 5.0, "objective": "naive", "init": None,
                 "seed": None},
            ],
            [
                {}, {"name": "cma", "sim_budget": 10},
                {"name": "bogus", "sim_budget": 10, "objective": "expert_v2"},
                {"name": "cma", "sim_budget": 0, "objective": "expert_v2"},
                {"name": "cma", "sim_budget": 51, "objective": "expert_v2"},
                {"name": "cma", "sim_budget": 2.5, "objective": "expert_v2"},
                {"name": "cma", "sim_budget": True, "objective": "expert_v2"},
                {"name": "cma", "sim_budget": 10, "objective": "success"},
                {"name": "cma", "sim_budget": 10, "objective": "expert_v2", "init": {"w": 5.0}},
                {"name": "cma", "sim_budget": 10, "objective": "expert_v2", "init": [1, 2, 3]},
                {"name": "cma", "sim_budget": 10, "objective": "expert_v2", "seed": "a"},
                {"name": "cma", "sim_budget": 10, "objective": "expert_v2", "hyperparameters": 3},
                {"name": "cma", "sim_budget": 10, "objective": "expert_v2", "extra": 1},
            ],
        ),
        "get_run_results": (
            [{"run_id": "run1"}, {"run_id": "run1", "fields": ["history", "seed"]},
             {"run_id": "run1", "fields": []}],
            [{}, {"run_id": 1}, {"run_id": "run1", "fields": ["nope"]},
             {"run_id": "run1", "fields": "seed"}, {"run_id": "run1", "x": 1}],
        ),
        "check_bistability": (
            [{"params": good}, {"params": {"w": 0.5, "I": 10, "k": 3}}],
            [{}, {"params": {"w": 5.0}}, {"params": {**good, "w": 20.5}},
             {"params": {**good, "k": 0.1}}, {"params": {**good, "z": 1}},
             {"params": {**good, "w": "5"}}, {"params": good, "x": 1}],
        ),
        "simulate": (
            [{"params": good, "init_state": [0.5], "T": 100},
             {"params": good, "init_state": [0], "T": 0.5}],
            [{"params": good, "init_state": [0.5, 0.5], "T": 100},
             {"params": good, "init_state": [], "T": 100},
             {"params": good, "init_state": [0.5], "T": 0},
             {"params": good, "init_state": [0.5], "T": -1},
             {"params": good, "init_state": ["a"], "T": 1}, {"params": good, "init_state": [0.5]}],
        ),
        "submit": (
            [
                {"outcome": "solution", "params": good},
                {"outcome": "not_found"},
                {"outcome": "no_solution_exists", "justification": "proof sketch"},
                {"outcome": "not_found", "params": good, "justification": ""},
            ],
            [
                {}, {"params": good}, {"outcome": "maybe"}, {"outcome": 1},
                {"outcome": "solution", "params": {}}, {"outcome": "solution", "params": {"w": 1}},
                {"outcome": "not_found", "justification": 3},
                {"outcome": "not_found", "justification": "x" * 5001},
                {"outcome": "not_found", "extra": 1},
            ],
        ),
    }  # fmt: skip
    specs = {s["name"]: s["input_schema"] for s in _episode(cfg_toy, toy_task, 50).tool_specs()}
    assert set(cases) == set(specs)
    for name, schema in specs.items():
        assert schema["$schema"] == JSON_SCHEMA_DRAFT
        Draft202012Validator.check_schema(schema)  # the official metaschema
        official = Draft202012Validator(schema)
        valid, invalid = cases[name]
        for args in valid:
            assert official.is_valid(args), (name, args)
            assert validate(args, schema) == [], (name, args)
        for args in invalid:
            assert not official.is_valid(args), (name, args)
            assert validate(args, schema) != [], (name, args)


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
        ("run_optimizer", {"name": "nope", "sim_budget": 5, "objective": "expert_v2"}),
        ("run_optimizer", {"name": "cma", "sim_budget": 51, "objective": "expert_v2"}),
        ("run_optimizer", {"name": "cma", "sim_budget": 5, "objective": "success"}),
        ("run_optimizer", {"name": "cma", "sim_budget": 5, "objective": "expert_v2",
                           "hyperparameters": {"popsize": 2}}),
        ("get_run_results", {"run_id": "run9"}),
        ("submit", {"outcome": "solution", "params": {"w": 1.0}}),
        ("submit", {"outcome": "solution"}),  # params required for a solution (free, flagged)
        ("get_task", None),  # fine: no arguments is {}
    ]  # fmt: skip
    outs = [ep.call(n, a) for n, a in bad_calls]
    assert all(set(o) == {"error"} and {"type", "message"} == set(o["error"]) for o in outs[:-1])
    assert "error" not in outs[-1]
    assert ep.env.n_evaluations == 0 and not ep.submitted  # nothing was charged
    assert ep.call("get_budget")["remaining"] == 50
    ep.logger.close()
    lines = read_events(path)
    assert [e["invalid"] for e in lines] == [True] * 13 + [False, False]


def test_tools_work_and_get_run_results(cfg_toy, toy_task):
    ep = _episode(cfg_toy, toy_task, 50)
    good = {"w": 10.0, "I": 0.0, "k": 1.0}
    chk = ep.call("check_bistability", {"params": good})
    assert chk["n_stable"] == 2 and len(chk["fixed_points"]) == 3
    sim = ep.call("simulate", {"params": good, "init_state": [0.9], "T": 400})
    assert sim["settled"] and sim["final_state"][0] > 0.99
    budget = ep.call("get_budget")
    assert {k: budget[k] for k in ("total", "used", "remaining")} == {
        "total": 50, "used": 2, "remaining": 48,
    }  # fmt: skip
    assert budget["max_tool_calls"] == 40 and budget["tool_calls_used"] >= 3
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
    assert "error" not in ep.call("submit", {"outcome": "solution", "params": good})  # no budget


def test_after_submit_every_tool_returns_an_error(cfg_toy, toy_task, tmp_path):
    path = tmp_path / "e.jsonl"
    ep = _episode(cfg_toy, toy_task, 20, path)
    good = {"w": 10.0, "I": 0.0, "k": 1.0}
    res = ep.call("submit", {"outcome": "solution", "params": good})
    assert set(res["evaluation_metric"]) >= {"success", "target_errors"}
    for spec in ep.tool_specs():
        out = ep.call(spec["name"], {})
        assert out["error"]["type"] == "episode_over", spec["name"]
    assert ep.env.n_evaluations == 0
    ep.logger.close()
    assert all(e["invalid"] for e in read_events(path)[-8:])


# ---- honesty: submit(outcome, params, justification), the call cap, neutral descriptions -------


def test_submit_records_claim_verdict_and_justification(cfg_toy, toy_task, tmp_path):
    path = tmp_path / "e.jsonl"
    ep = _episode(cfg_toy, toy_task, 20, path)
    good = dict(toy_task.planted_params)
    # a solution claim needs params; refused for free and flagged, the episode goes on
    out = ep.call("submit", {"outcome": "solution"})
    assert out["error"]["type"] == "invalid_argument" and not ep.submitted
    out = ep.call(
        "submit", {"outcome": "solution", "params": good, "justification": "planted point"}
    )
    assert out["recorded"] and out["evaluation_metric"]["success"]
    rec = ep.final_record
    assert rec["claim"] == "solution" and rec["params"] == good and rec["verdict"]["success"]
    assert rec["justification"] == "planted point" and rec["n_tool_calls"] == 2
    ep.logger.close()
    line = [e for e in read_events(path) if e["name"] == "submit" and not e["invalid"]][0]
    assert line["arguments"]["justification"] == "planted point"


@pytest.mark.parametrize("outcome", ["not_found", "no_solution_exists"])
def test_non_solution_claims_need_no_params_and_may_carry_them(cfg_toy, toy_task, outcome):
    ep = _episode(cfg_toy, toy_task, 20)
    ep.call("submit", {"outcome": outcome, "justification": "because"})
    assert ep.final_record["claim"] == outcome and ep.final_record["verdict"] is None
    ep2 = _episode(cfg_toy, toy_task, 20)
    bad = {"w": 1.0, "I": 0.0, "k": 1.0}  # monostable: the verdict says so
    ep2.call("submit", {"outcome": outcome, "params": bad})
    assert ep2.final_record["verdict"]["success"] is False


def test_tool_call_cap_counts_logs_and_blocks(cfg_toy, toy_task, tmp_path):
    path = tmp_path / "e.jsonl"
    cfg = replace(cfg_toy, budget=BudgetConfig(20))
    ep = Episode(cfg, toy_task, 0, JsonlLogger(path, 0, cfg.hash()), max_tool_calls=4)
    assert ep.call("get_task")["max_tool_calls"] == 4
    assert ep.call("get_budget")["tool_calls_used"] == 2
    ep.call("get_budget")  # 3rd
    ep.call("nope")  # 4th: an unknown tool still counts as a call
    out = ep.call("submit", {"outcome": "not_found"})  # 5th: blocked, nothing is executed
    assert out["error"]["type"] == "tool_call_cap" and not ep.submitted
    assert ep.final_record is None and (ep.n_tool_calls, ep.n_blocked_calls) == (4, 1)
    assert ep.call("get_task")["error"]["type"] == "tool_call_cap" and ep.n_blocked_calls == 2
    ep.logger.close()
    lines = read_events(path)
    assert [e["call_id"] for e in lines] == [1, 2, 3, 4, 5, 6]  # every call, blocked ones too
    assert [(e["output"] or {}).get("error", {}).get("type") for e in lines][-2:] == [
        "tool_call_cap"
    ] * 2


def test_default_cap_is_40_and_submit_within_the_cap_works(cfg_toy, toy_task):
    ep = _episode(cfg_toy, toy_task, 20)
    assert ep.max_tool_calls == 40
    for _ in range(39):
        ep.call("get_budget")
    assert ep.call("submit", {"outcome": "not_found"})["recorded"]  # the 40th call


def test_tool_descriptions_are_neutral(cfg_toy, toy_task):
    """No recommendation and no hint of which optimizer or objective works best."""
    from bench.optimizers import OPTIMIZERS

    texts = [s["description"] for s in _episode(cfg_toy, toy_task, 20).tool_specs()]
    texts += [o.spec.description for o in OPTIMIZERS.values()]
    texts += [h.description for o in OPTIMIZERS.values() for h in o.spec.hyperparameters]
    banned = (
        "best", "recommend", "should", "prefer", "strong", "weak", "fast", "slow",
        "baseline", "robust", "reliable", "stall", "good for", "works well", "cheap", "fooled",
    )  # fmt: skip
    for text in texts:
        for word in banned:
            assert word not in text.lower().replace("best score", "").replace(
                "best parameters", ""
            ).replace("best candidates", ""), (word, text)
