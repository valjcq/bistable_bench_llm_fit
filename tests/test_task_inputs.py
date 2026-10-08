"""Reference range, tol_fraction and task inputs (the Wilson-Cowan P), and their back-compat."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bench.checker import check
from bench.config import config_from_dict, load_config
from bench.env import Environment
from bench.tasks import Task, draw_task_inputs, load_task, make_task
from bench.tools import Episode

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg_wc():
    return load_config(ROOT / "configs/exp_wc1972.yaml")


def test_tol_is_a_fraction_of_the_reference_range(cfg_wc, cfg_toy, cfg_ei):
    assert cfg_wc.model().reference_range == (0.0, 0.5)
    assert cfg_wc.task.tol_fraction == 0.02 and cfg_wc.task.tol == pytest.approx(0.01)
    assert cfg_wc.checker.tol == cfg_wc.task.tol  # the checker uses the derived value
    for cfg in (cfg_toy, cfg_ei):  # logistic models: range [0, 1], tol 0.02 as before
        assert cfg.model().reference_range == (0.0, 1.0) and cfg.task.tol == 0.02


def test_tol_and_tol_fraction_must_agree(cfg_wc):
    with pytest.raises(ValueError, match="tol_fraction"):
        replace(
            cfg_wc, task=replace(cfg_wc.task, tol=0.02), checker=replace(cfg_wc.checker, tol=0.02)
        )
    with pytest.raises(ValueError, match="not both"):
        config_from_dict(
            {"task": {"tol": 0.01, "tol_fraction": 0.02}}, cfg_wc.model_spec, resolved=False
        )


def test_resolved_config_round_trips_and_old_configs_do_not_mention_tol_fraction(
    cfg_wc, cfg_toy, cfg_ei
):
    again = config_from_dict(cfg_wc.resolved(), cfg_wc.model_spec, resolved=True)
    assert again.hash() == cfg_wc.hash() and again.task.tol == cfg_wc.task.tol
    for cfg in (cfg_toy, cfg_ei):  # a default-valued new field must not enter the hash
        assert "tol_fraction" not in cfg.resolved()["task"]
        assert "reference_range" not in cfg.model_spec and "task_inputs" not in cfg.model_spec


def test_committed_logistic_tasks_have_no_inputs_and_keep_their_format():
    task = load_task(ROOT / "tasks/toy_1pop_K2_seed0.json")
    assert task.inputs == {} and "inputs" not in task.to_dict()
    raw = json.loads((ROOT / "tasks/toy_1pop_K2_seed0.json").read_text())
    assert task.to_dict() == raw


def test_draw_task_inputs_respects_the_range_and_the_minimum(cfg_wc, cfg_toy):
    model = cfg_wc.model()
    rng = np.random.default_rng(0)
    values = [draw_task_inputs(model, rng)["P"] for _ in range(2000)]
    assert min(values) >= -0.3 and max(values) <= 0.3
    assert min(abs(v) for v in values) >= 0.05
    assert min(values) < -0.25 and max(values) > 0.25  # both signs, full range
    # models without task inputs draw nothing and do not touch the random stream
    rng2, rng3 = np.random.default_rng(1), np.random.default_rng(1)
    assert draw_task_inputs(cfg_toy.model(), rng2) == {}
    assert rng2.random() == rng3.random()


def test_environment_applies_the_task_input_and_tells_the_agent(cfg_wc):
    cfg = replace(cfg_wc)
    model = cfg.model()
    params = {
        "c1": 12, "c2": 4, "c3": 13, "c4": 11, "a_e": 1.2, "theta_e": 2.8, "a_i": 1.0,
        "theta_i": 4.0,
    }  # fmt: skip
    targets = np.array([[0.1], [0.4]])

    def task_with(P: float) -> Task:
        return Task("wc1972", ("E",), 0, 2, targets, params, cfg.generation_hash(), 1, {}, {"P": P})

    env_a, env_b = (Environment.from_task(cfg, task_with(P)) for P in (0.0 + 0.1, -0.2))
    assert env_a.describe()["inputs"] == {"P": 0.1}
    ra, rb = env_a.check(params), env_b.check(params)
    assert ra.to_dict() != rb.to_dict()  # the input changes the fixed points
    direct = check(model.with_inputs({"P": 0.1}), params, targets, cfg.checker)
    assert ra.to_dict() == direct.to_dict()
    for bad in ({}, {"P": 0.01}, {"P": 0.5}, {"P": 0.1, "Q": 0.0}):
        with pytest.raises(ValueError):
            model.with_inputs(bad)
    assert (
        "inputs"
        not in Environment.from_task(
            load_config(ROOT / "configs/exp_toy_1pop.yaml"),
            load_task(ROOT / "tasks/toy_1pop_K2_seed0.json"),
        ).describe()
    )


def test_a_wc_task_can_be_planted_with_its_input_and_is_told_to_the_agent(cfg_wc):
    """In-memory only (no task file): a relaxed window, because with the 5-95% window of
    [0, 0.5] almost no random draw qualifies (scripts/wc_feasibility.py)."""
    cfg = replace(cfg_wc, task=replace(cfg_wc.task, target_window=(0.0, 1.0), max_attempts=3000))
    task = make_task(cfg, seed=3)
    again = make_task(cfg, seed=3)
    assert task.to_dict() == again.to_dict()  # deterministic, input included
    P = task.inputs["P"]
    assert 0.05 <= abs(P) <= 0.3 and set(task.planted_params) == set(cfg.model().space.names)
    env = Environment.from_task(cfg, task)
    assert env.check(task.planted_params).success  # solvable at the planted input
    info = Episode(cfg, task, seed=0).call("get_task")
    assert info["inputs"] == {"P": P} and info["tol"] == pytest.approx(0.01)
    assert "planted" not in json.dumps(info)


# ---- asymmetric window (lowest target free) and corner starts --------------------------------


def test_lowest_target_free_is_omitted_from_old_hashes_and_validated(cfg_wc, cfg_toy):
    assert cfg_wc.task.lowest_target_free and "lowest_target_free" in cfg_wc.resolved()["task"]
    assert "lowest_target_free" not in cfg_toy.resolved()["task"]  # default: not in the hash
    from bench.tasks import pick_targets

    rng = np.random.default_rng(0)
    lo, hi = np.array([0.05]), np.array([0.45])
    with pytest.raises(NotImplementedError):
        pick_targets([np.array([0.1])], 3, lo, hi, 0.03, True, rng)


def test_pick_targets_low_is_free_and_high_must_be_in_the_window():
    from bench.tasks import pick_targets

    rng = np.random.default_rng(0)
    lo, hi = np.array([0.05]), np.array([0.45])
    v = lambda *xs: [np.array([x]) for x in xs]  # noqa: E731
    got = pick_targets(v(0.0, 0.3), 2, lo, hi, 0.03, True, rng)
    assert [float(g[0]) for g in got] == [0.0, 0.3]  # low is outside the window: allowed
    assert pick_targets(v(0.0, 0.5), 2, lo, hi, 0.03, True, rng) is None  # high saturated
    assert pick_targets(v(0.28, 0.3), 2, lo, hi, 0.03, True, rng) is None  # too close
    both = pick_targets(v(0.1, 0.4), 2, lo, hi, 0.03, True, rng)
    assert [float(g[0]) for g in both] == [0.1, 0.4]  # an in-window state may be the low one
    assert pick_targets(v(0.0, 0.3), 2, lo, hi, 0.03, False, rng) is None  # normal rule: both


def test_wc_planting_with_the_adopted_window_is_cheap_and_solvable(cfg_wc):
    task = make_task(cfg_wc, seed=5)
    assert task.n_attempts < 1000 and 0.05 <= abs(task.inputs["P"]) <= 0.3
    low, high = task.targets.ravel()
    assert 0.05 <= high <= 0.45 and high - low > 3 * cfg_wc.task.tol
    assert Environment.from_task(cfg_wc, task).check(task.planted_params).success


def test_corner_starts_find_the_saturated_states_with_a_single_random_start(wc_free_p):
    """With ONE random start the box corners still find the quiescent and the saturated stable
    states of the paper's Fig. 4 set (0, 0) and (0.44, 0.226) (found without a target)."""
    from bench.checker import CheckerConfig, find_stable_states
    from bench.models.wc1972 import free_params, paper_sets

    s = paper_sets()["fig4_5"]
    stable = find_stable_states(wc_free_p, {**free_params(s), "P": 0.0}, CheckerConfig(n_starts=1))
    found = sorted(round(fp.r[0], 3) for fp in stable)
    assert found == [0.0, 0.44]


@pytest.fixture(scope="module")
def wc_free_p(cfg_wc):
    from bench.models.wc1972 import with_free_input

    return with_free_input(cfg_wc.model())


def test_committed_wc_task_files_match_the_current_config(cfg_wc):
    for f in sorted((ROOT / "tasks").glob("wc1972_K2_seed*.json")) + sorted(
        (ROOT / "tasks/selection").glob("wc1972_K2_seed*.json")
    ):
        task = load_task(f)
        assert task.generation_hash == cfg_wc.generation_hash(), f"{f.name}: stale"
        assert Environment.from_task(cfg_wc, task).check(task.planted_params).success
        assert 0.05 <= abs(task.inputs["P"]) <= 0.3
