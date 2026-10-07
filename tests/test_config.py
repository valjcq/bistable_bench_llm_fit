"""Config loading, validation and hashing."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from bench.config import BudgetConfig, ExperimentConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path, extra: str = "", model: str = "toy_1pop.yaml") -> str:
    """Write an experiment YAML next to a copy of the toy model file."""
    (tmp_path / "toy_1pop.yaml").write_text((CONFIGS / "toy_1pop.yaml").read_text())
    path = tmp_path / "exp.yaml"
    path.write_text(f"model: {model}\n{extra}")
    return str(path)


def test_defaults_fill_missing_sections(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.task.K == 2 and cfg.task.tol == 0.02 and cfg.budget.max_evaluations == 200
    assert cfg.checker.tol == cfg.task.tol  # taken from the task section


@pytest.mark.parametrize(
    "extra",
    [
        "task: {tolerance: 0.1}\n",  # typo -> unknown key
        "checker: {tol: 0.5}\n",  # tol belongs to `task`
        "bogus: 1\n",
        "task: {K: 1}\n",
        "task: {target_window: [0.9, 0.1]}\n",
        "budget: {max_evaluations: 0}\n",
        "scoring: {naive_ic_fractions: [0.0, 0.5, 1.0]}\n",  # K=2 needs 2 entries
        "scoring: {naive_ic_fractions: [0.0, 1.5]}\n",
        "simulator: {dt: -1.0}\n",
    ],
)
def test_invalid_configs_are_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, extra))


def test_hash_is_stable_and_sensitive(cfg_toy, cfg_ei):
    assert cfg_toy.hash() == load_config(CONFIGS / "exp_toy_1pop.yaml").hash()
    assert cfg_toy.hash() != cfg_ei.hash()
    more_budget = replace(cfg_toy, budget=BudgetConfig(999))
    assert more_budget.hash() != cfg_toy.hash()
    # ... but the budget does not affect how tasks are generated, so that hash is unchanged:
    assert more_budget.generation_hash() == cfg_toy.generation_hash()
    more_attempts = replace(cfg_toy, task=replace(cfg_toy.task, max_attempts=1))
    assert more_attempts.generation_hash() == cfg_toy.generation_hash()  # only a give-up limit
    changed_tol = replace(
        cfg_toy,
        task=replace(cfg_toy.task, tol=0.05),
        checker=replace(cfg_toy.checker, tol=0.05),
    )
    with pytest.raises(ValueError):  # the two copies of tol must agree
        replace(cfg_toy, task=replace(cfg_toy.task, tol=0.05))
    assert changed_tol.generation_hash() != cfg_toy.generation_hash()


def test_hash_covers_model_content_not_its_path(tmp_path, cfg_toy):
    spec = dict(cfg_toy.model_spec)
    spec["fixed"] = {**spec["fixed"], "tau": 11.0}
    other = ExperimentConfig(
        model_spec=spec,
        task=cfg_toy.task,
        checker=cfg_toy.checker,
        simulator=cfg_toy.simulator,
        scoring=cfg_toy.scoring,
        budget=cfg_toy.budget,
    )
    assert other.hash() != cfg_toy.hash()
    assert yaml.safe_load(open(_write(tmp_path)))["model"] == "toy_1pop.yaml"
