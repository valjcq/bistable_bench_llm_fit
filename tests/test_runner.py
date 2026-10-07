"""Checkpoint 6: logging, runner, batch summary."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bench.batch import (
    final_error,
    first_success_eval,
    format_summary,
    parse_conditions,
    parse_seeds,
    run_batch,
    summarize,
    wilson_interval,
)
from bench.config import BaselineConfig, BudgetConfig, config_from_dict
from bench.env import Environment
from bench.logging import VOLATILE_FIELDS, JsonlLogger, comparable, read_events
from bench.runner import choose_submission, run_baseline
from bench.tasks import load_task

TASKS = Path(__file__).resolve().parents[1] / "tasks"
FIELDS = {
    "timestamp", "step", "role", "name", "arguments", "output", "output_sha256", "tokens",
    "latency", "budget_after", "seed", "config_hash", "invalid", "call_id", "parent_call_id",
}  # fmt: skip


@pytest.fixture
def small_cfg(cfg_toy):
    return replace(cfg_toy, budget=BudgetConfig(40))


@pytest.fixture
def toy_task():
    return load_task(TASKS / "toy_1pop_K2_seed4.json")


def _run(cfg, task, cond, seed, out, opt=None):
    return run_baseline(cfg, task, "toy_1pop_K2_seed4", cond, seed, out, opt)


@pytest.mark.parametrize(("cond", "opt"), [("A", None), ("B", "cma"), ("C", "tpe")])
def test_run_writes_a_complete_log(small_cfg, toy_task, tmp_path, cond, opt):
    res = _run(small_cfg, toy_task, cond, 1, tmp_path, opt)
    d = tmp_path / res["run_name"]
    assert {p.name for p in d.iterdir()} == {"events.jsonl", "config.json", "result.json"}
    events = read_events(d / "events.jsonl")
    assert all(set(e) == FIELDS for e in events)
    assert [e["step"] for e in events] == list(range(len(events)))
    names = [e["name"] for e in events]
    # submission rule: 3 evaluations reserved from the 40, spent on checker calls at the end
    assert names.count("evaluate") == 37 and names.count("check") == 3 == res["evals_used"] - 37
    assert names[-6:] == ["run_optimizer", "check", "check", "check", "select_submission",
                          "final_report"]  # fmt: skip
    assert {e["role"] for e in events if e["name"] == "check"} == {"harness"}
    assert events[-1]["role"] == "harness"
    # the budget counts down by one per evaluation and ends at zero
    ev = [e for e in events if e["name"] == "evaluate"]
    assert [e["budget_after"] for e in ev] == list(range(39, 2, -1))  # 3 stay reserved
    assert {e["seed"] for e in events} == {1} and len({e["config_hash"] for e in events}) == 1
    assert all(e["tokens"] is None and e["latency"] is None for e in events)
    result = json.loads((d / "result.json").read_text())
    assert result["evaluation_metric"] == events[-1]["output"]["metric"]
    # planted parameters appear nowhere in the run directory
    planted = json.dumps(toy_task.planted_params)
    assert all(planted not in p.read_text() for p in d.iterdir())


def test_rerun_is_identical_except_volatile_fields(small_cfg, toy_task, tmp_path):
    a = _run(small_cfg, toy_task, "B", 3, tmp_path / "a", "tpe")
    b = _run(small_cfg, toy_task, "B", 3, tmp_path / "b", "tpe")
    ea = read_events(tmp_path / "a" / a["run_name"] / "events.jsonl")
    eb = read_events(tmp_path / "b" / b["run_name"] / "events.jsonl")
    assert comparable(ea) == comparable(eb)
    assert ea != eb  # the timestamps do differ, so the comparison is not vacuous
    for r in (a, b):
        r.pop("wall_clock_s")
    assert a == b
    c = _run(small_cfg, toy_task, "B", 4, tmp_path / "c", "tpe")  # another seed differs
    ec = read_events(tmp_path / "c" / c["run_name"] / "events.jsonl")
    assert comparable(ec) != comparable(ea)


@pytest.mark.parametrize(("cond", "opt"), [("A", None), ("B", "cma"), ("C", "cma")])
def test_run_is_reconstructible_from_events_and_config(small_cfg, toy_task, tmp_path, cond, opt):
    """Rebuild the environment from config.json and replay the log: same scores, same verdict."""
    res = _run(small_cfg, toy_task, cond, 2, tmp_path, opt)
    d = tmp_path / res["run_name"]
    saved = json.loads((d / "config.json").read_text())
    exp = saved["experiment"]
    cfg = config_from_dict(exp, exp["model"], resolved=True)
    assert cfg.hash() == res["config_hash"]  # the saved config is the one that was run
    env = Environment(cfg, np.array(saved["task"]["targets"]), saved["task"]["seed"])
    all_events = read_events(d / "events.jsonl")
    run_line = next(e for e in all_events if e["name"] == "run_optimizer")
    ev = [e for e in all_events if e["name"] == "evaluate"]
    assert {e["parent_call_id"] for e in ev} == {run_line["call_id"]}  # grouped by call
    for e in ev:  # every logged score is reproduced by a fresh environment
        assert env.evaluate(e["arguments"]["params"], saved["objective"]) == e["output"]["value"]
    sel = next(e for e in all_events if e["name"] == "select_submission")
    ranked = sorted(ev, key=lambda e: e["output"]["value"])  # stable: earliest first on ties
    top = list({tuple(sorted(e["arguments"]["params"].items())): e for e in ranked}.values())[:3]
    chosen = top[sel["output"]["chosen_rank"]]  # the rule's pick, from the logged ranking
    assert [e["output"]["value"] for e in top] == sel["output"]["scores"]
    metric = env.final_report(chosen["arguments"]["params"])
    assert metric.success == res["evaluation_metric"]["success"]
    assert metric.target_errors == res["evaluation_metric"]["target_errors"]


def test_choose_submission_rule():
    a, b, c, d = ({"x": float(i)} for i in range(4))
    scored = [(0.5, c), (0.1, a), (0.1, a), (0.3, b), (0.9, d), (0.1, a)]  # a evaluated 3 times
    ok = {0.0: False, 1.0: True, 2.0: True, 3.0: True}  # pass/fail by params["x"]
    passes = lambda p: ok[p["x"]]  # noqa: E731
    # top-3 DISTINCT by score: a (0.1), b (0.3), c (0.5); a fails, b passes -> b
    chosen, rep = choose_submission(scored, 3, passes)
    assert chosen == b and rep["scores"] == [0.1, 0.3, 0.5] and rep["passed"] == [False, True, True]
    assert rep["chosen_rank"] == 1
    # d would pass but is outside the top 3: never considered
    ok.update({1.0: False, 2.0: False})
    chosen, rep = choose_submission(scored, 3, passes)
    assert chosen == a and rep["chosen_rank"] == 0 and rep["passed"] == [False] * 3  # none: best
    chosen, rep = choose_submission(scored, 0, passes)  # k = 0: the old rule, no checks at all
    assert chosen == a and rep["passed"] == []
    assert choose_submission([(0.2, a)], 3, passes)[1]["scores"] == [0.2]  # fewer than k


def test_k0_reproduces_the_old_submission_rule(small_cfg, toy_task, tmp_path):
    cfg0 = replace(small_cfg, baseline=replace(small_cfg.baseline, submission_checks=0))
    res = _run(cfg0, toy_task, "C", 0, tmp_path)
    ev = [e for e in read_events(tmp_path / res["run_name"] / "events.jsonl")]
    assert [e["name"] for e in ev].count("evaluate") == 40 == res["evals_used"]
    assert res["submission"]["passed"] == [] and cfg0.hash() != small_cfg.hash()


def test_large_outputs_are_replaced_by_their_hash(tmp_path):
    with JsonlLogger(tmp_path / "e.jsonl", 0, "h", max_output_chars=50) as lg:
        lg.log("x", "small", {}, {"a": 1}, 5)
        lg.log("x", "big", {}, {"a": "y" * 100}, 5)
    small, big = read_events(tmp_path / "e.jsonl")
    assert small["output"] == {"a": 1} and small["output_sha256"] is None
    assert big["output"] is None and len(big["output_sha256"]) == 64
    assert set(VOLATILE_FIELDS) == {"timestamp", "latency"}


def test_wilson_interval_matches_known_values():
    lo, hi = wilson_interval(5, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.237, 0.763)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert math.isnan(wilson_interval(0, 0)[0])


def test_first_success_expert_and_naive(cfg_toy, tmp_path):
    """Expert needs no checker call; a fooled naive score (<= tol but failing) is skipped."""
    task = load_task(TASKS / "toy_1pop_K2_seed4.json")

    def events(objective, items):
        with JsonlLogger(tmp_path / f"{objective}.jsonl", 0, "h") as lg:
            for i, (params, value) in enumerate(items, start=1):
                args = {"params": params, "objective": objective}
                lg.log("b", "evaluate", args, {"value": value}, 200 - i)
        return tmp_path / f"{objective}.jsonl"

    good, bad = task.planted_params, {"w": 1.0, "I": 0.0, "k": 1.0}  # monostable: fails
    # expert_v2 scores are not on the tol scale: every point is a candidate, the checker decides
    p = events("expert_v2", [(bad, 0.7), (bad, 0.01), (good, 0.9)])
    assert first_success_eval(cfg_toy, task, p) == 3  # the successful point has a HIGH score
    p = events("naive", [(bad, 0.7), (bad, 0.01), (good, 0.015)])
    assert first_success_eval(cfg_toy, task, p) == 3  # naive: point 2 is checked and fails
    p = events("naive", [(bad, 0.7), (bad, 0.01)])
    assert first_success_eval(cfg_toy, task, p) is None


def test_override_optimizer_is_recorded_in_config_and_hash(cfg_toy, small_cfg, toy_task, tmp_path):
    assert small_cfg.baseline.optimizer == "cma"
    res = _run(small_cfg, toy_task, "C", 0, tmp_path, "tpe")
    saved = json.loads((tmp_path / res["run_name"] / "config.json").read_text())
    assert saved["optimizer"] == "tpe" and saved["experiment"]["baseline"]["optimizer"] == "tpe"
    assert res["optimizer"] == "tpe" and "C-tpe" in res["run_name"]
    expected = replace(small_cfg, baseline=BaselineConfig("tpe")).hash()
    assert res["config_hash"] == expected != small_cfg.hash()
    first = read_events(tmp_path / res["run_name"] / "events.jsonl")[0]
    assert first["config_hash"] == expected
    with pytest.raises(ValueError):
        _run(small_cfg, toy_task, "A", 0, tmp_path, "tpe")  # A is always random search


def test_batch_summary_and_parsing(small_cfg, tmp_path):
    assert parse_seeds("0-2,5") == [0, 1, 2, 5]
    assert parse_conditions(["A", "B:cma"]) == [("A", None), ("B", "cma")]
    assert parse_conditions(["S"]) == [("S", None)]
    for bad in ("A:cma", "B", "D:cma", "S:cma"):
        with pytest.raises(ValueError):
            parse_conditions([bad])
    rows = run_batch(
        small_cfg,
        [TASKS / "toy_1pop_K2_seed4.json"],
        parse_conditions(["A", "C:cma"]),
        [0, 1],
        tmp_path,
    )
    assert len(rows) == 4 and {r["label"] for r in rows} == {"A", "C-cma"}
    s = summarize(rows)
    assert [x["runs"] for x in s] == [2, 2]
    assert "| A | 2 |" in format_summary(s)
    assert final_error({"target_errors": [0.1, None]}) == pytest.approx(0.55)


def test_parallel_batch_matches_serial(small_cfg, tmp_path):
    args = (small_cfg, [TASKS / "toy_1pop_K2_seed4.json"], parse_conditions(["C:cma", "S"]), [0, 1])
    serial = run_batch(*args, tmp_path / "s")
    parallel = run_batch(*args, tmp_path / "p", jobs=2)
    drop = lambda rows: [{k: v for k, v in r.items() if k != "wall_clock_s"} for r in rows]  # noqa: E731
    assert drop(serial) == drop(parallel)


def test_condition_c_uses_expert_v2_and_confirms_every_point(small_cfg, toy_task, tmp_path):
    res = _run(small_cfg, toy_task, "C", 0, tmp_path, "cma")
    d = tmp_path / res["run_name"]
    assert res["objective"] == "expert_v2" and "C-cma" in res["run_name"]
    ev = [e for e in read_events(d / "events.jsonl") if e["name"] == "evaluate"]
    assert {e["arguments"]["objective"] for e in ev} == {"expert_v2"} and len(ev) == 37
    # expert_v2 scores are not on the tol scale: first success must not assume score <= tol
    first_success_eval(small_cfg, toy_task, d / "events.jsonl")  # runs without error
