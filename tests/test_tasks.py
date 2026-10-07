"""Test 4: planted tasks are solvable, plus the generation guards and the saved task files."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bench.checker import check
from bench.config import load_config
from bench.tasks import TaskGenerationError, load_task, make_task, save_task

TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _min_pairwise_distance(targets: np.ndarray) -> float:
    return min(
        np.abs(targets[i] - targets[j]).max()
        for i in range(len(targets))
        for j in range(i + 1, len(targets))
    )


@pytest.mark.parametrize("which", ["toy", "ei"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_planted_params_solve_their_task(which, seed, cfg_toy, cfg_ei):
    cfg = cfg_toy if which == "toy" else cfg_ei
    task = make_task(cfg, seed=seed)
    model = cfg.model()
    res = check(model, task.planted_params, task.targets, cfg.checker)
    assert res.success and all(res.target_matched)
    assert task.targets.shape == (cfg.task.K, len(model.observed))
    # Guards: targets inside the window (fractions of r_max = 1) and farther apart than 3 * tol.
    lo, hi = cfg.task.target_window
    assert task.targets.min() >= lo and task.targets.max() <= hi
    assert _min_pairwise_distance(task.targets) > cfg.task.min_separation_tol_factor * cfg.task.tol
    # Sorted by increasing mean rate (the naive objective relies on this order).
    assert list(task.targets.mean(axis=1)) == sorted(task.targets.mean(axis=1))


def test_planted_task_with_noise_is_still_solvable(cfg_toy):
    cfg = replace(cfg_toy, task=replace(cfg_toy.task, noise=0.005))
    task = make_task(cfg, seed=1)
    clean = make_task(cfg_toy, seed=1)
    assert not np.array_equal(task.targets, clean.targets)  # noise was actually added
    assert check(cfg.model(), task.planted_params, task.targets, cfg.checker).success


def test_separation_guard_is_configurable(cfg_toy):
    cfg = replace(cfg_toy, task=replace(cfg_toy.task, min_separation_tol_factor=20.0))
    task = make_task(cfg, seed=0)  # targets must now be > 0.4 apart
    assert _min_pairwise_distance(task.targets) > 0.4


def test_generation_failure_is_a_clear_error(cfg_toy):
    cfg = replace(cfg_toy, task=replace(cfg_toy.task, max_attempts=3))
    with pytest.raises(TaskGenerationError) as err:
        make_task(cfg, seed=0)
    msg = str(err.value)
    assert "3 random draws" in msg and "toy_1pop" in msg and "fewer_than_K" in msg


def test_save_load_roundtrip(tmp_path, task_toy):
    save_task(task_toy, tmp_path / "t.json")
    again = load_task(tmp_path / "t.json")
    assert again.to_dict() == task_toy.to_dict()  # floats survive JSON exactly


@pytest.mark.parametrize("name", ["toy_1pop", "ei_2pop"])
def test_committed_task_files_are_valid_for_current_configs(name):
    """tasks/*.json were generated once; they must still match the configs and be solvable."""
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / f"exp_{name}.yaml")
    files = sorted(TASK_DIR.glob(f"{name}_K2_seed*.json"))
    assert files, f"no committed tasks for {name}"
    for f in files:
        task = load_task(f)
        assert task.generation_hash == cfg.generation_hash(), f"{f.name}: stale task file"
        assert check(cfg.model(), task.planted_params, task.targets, cfg.checker).success
