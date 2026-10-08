"""model_C's agent-facing parameter names: a mapping at the tool layer only."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from bench.config import BudgetConfig, load_config
from bench.logging import JsonlLogger, read_events
from bench.models.instances import PUBLIC_PARAM_NAMES, public_param_names
from bench.tasks import load_task
from bench.tools import Episode

ROOT = Path(__file__).resolve().parents[1]
MAP = {"c1": "w_EE", "c2": "w_EI", "c3": "w_IE", "c4": "w_II", "a_e": "gain_E",
       "theta_e": "thresh_E", "a_i": "gain_I", "theta_i": "thresh_I"}  # fmt: skip
INTERNAL = re.compile(r"\b(c[1-4]|a_e|theta_e|a_i|theta_i)\b")


@pytest.fixture(scope="module")
def cfg():
    return replace(load_config(ROOT / "configs/exp_wc1972.yaml"), budget=BudgetConfig(40))


@pytest.fixture(scope="module")
def task():
    return load_task(ROOT / "tasks/honesty/wc1972_K2_seed401.json")  # feasible


def test_the_mapping_is_the_requested_one_and_covers_every_free_parameter(cfg):
    assert public_param_names("wc1972") == MAP == PUBLIC_PARAM_NAMES["wc1972"]
    assert public_param_names("wc1972_k1") == MAP
    assert set(MAP) == set(cfg.model().space.names) and len(set(MAP.values())) == 8
    assert public_param_names("toy_1pop") == {} and public_param_names("ei_2pop") == {}


def test_get_task_and_the_schemas_use_public_names(cfg, task):
    ep = Episode(cfg, task, 0)
    names = [p["name"] for p in ep.call("get_task")["free_parameters"]]
    assert names == [MAP[n] for n in cfg.model().space.names]
    specs = {s["name"]: s["input_schema"] for s in ep.tool_specs()}
    for tool in ("check_bistability", "simulate", "submit"):
        props = specs[tool]["properties"]["params"]
        assert list(props["properties"]) == names and props["required"] == names
    assert list(specs["run_optimizer"]["properties"]["init"]["properties"]) == names


def test_tools_accept_public_names_and_return_them(cfg, task):
    ep = Episode(cfg, task, 0)
    public = {MAP[k]: v for k, v in task.planted_params.items()}
    chk = ep.call("check_bistability", {"params": public})
    assert chk["success"] is True  # the planted solution, in the agent's vocabulary
    run = ep.call(
        "run_optimizer",
        {"name": "random", "sim_budget": 5, "objective": "expert_v2", "init": public},
    )
    assert set(run["best_params"]) == set(MAP.values())
    assert run["best_score"] == pytest.approx(0.0, abs=1e-8)  # init was evaluated first
    got = ep.call("get_run_results", {"run_id": "run1", "fields": ["best_params", "history"]})
    assert set(got["best_params"]) == set(MAP.values())
    sim = ep.call("simulate", {"params": public, "init_state": [0.1, 0.1], "T": 50})
    assert "final_state" in sim
    done = ep.call("submit", {"outcome": "solution", "params": public})
    assert done["evaluation_metric"]["success"] is True
    assert set(ep.final_record["params"]) == set(MAP.values())


def test_internal_names_are_refused_and_nothing_is_charged(cfg, task):
    ep = Episode(cfg, task, 0)
    out = ep.call("check_bistability", {"params": task.planted_params})
    assert out["error"]["type"] == "invalid_argument" and ep.env.n_evaluations == 0


def test_error_messages_are_translated(cfg, task):
    ep = Episode(cfg, task, 0)
    text = ep._public_text("expected free params ['a_e', 'c1', 'theta_i'], got ['c2'] and c10 a_e2")
    assert text == "expected free params ['gain_E', 'w_EE', 'thresh_I'], got ['w_EI'] and c10 a_e2"
    toy = Episode(load_config(ROOT / "configs/exp_toy_1pop.yaml"),
                  load_task(ROOT / "tasks/honesty/toy_1pop_K2_seed302.json"), 0)  # fmt: skip
    assert toy._public_text("w and k") == "w and k"  # models without a mapping: unchanged


def test_internal_names_stay_in_task_files_and_the_analysis_side_of_the_log(cfg, task, tmp_path):
    assert set(task.planted_params) == set(MAP)  # task files unchanged
    path = tmp_path / "e.jsonl"
    ep = Episode(cfg, task, 0, JsonlLogger(path, 0, cfg.hash()))
    public = {MAP[k]: v for k, v in task.planted_params.items()}
    ep.call("get_task")
    ep.call("check_bistability", {"params": public})
    ep.call("run_optimizer", {"name": "random", "sim_budget": 3, "objective": "naive"})
    ep.logger.close()
    for e in read_events(path):
        text = json.dumps([e["arguments"], e["output"]])
        if e["call_id"] is not None:  # a tool-call line: what the agent sent and received
            assert not INTERNAL.search(text), e["name"]
    # the environment's own event lines (never shown to the agent) keep the internal names
    env_lines = [e for e in read_events(path) if e["call_id"] is None and e["name"] == "check"]
    assert env_lines and set(env_lines[0]["arguments"]["params"]) == set(MAP)
