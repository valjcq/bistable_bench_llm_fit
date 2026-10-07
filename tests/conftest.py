from pathlib import Path

import pytest

from bench.config import ExperimentConfig, load_config
from bench.models.instances import load_model
from bench.models.rate_network import RateNetwork
from bench.tasks import make_task

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def toy() -> RateNetwork:
    return load_model(CONFIGS / "toy_1pop.yaml")


@pytest.fixture(scope="session")
def ei() -> RateNetwork:
    return load_model(CONFIGS / "ei_2pop.yaml")


@pytest.fixture(scope="session")
def cfg_toy() -> ExperimentConfig:
    return load_config(CONFIGS / "exp_toy_1pop.yaml")


@pytest.fixture(scope="session")
def cfg_ei() -> ExperimentConfig:
    return load_config(CONFIGS / "exp_ei_2pop.yaml")


@pytest.fixture(scope="session")
def task_toy(cfg_toy):
    return make_task(cfg_toy, seed=0)


@pytest.fixture(scope="session")
def task_ei(cfg_ei):
    return make_task(cfg_ei, seed=0)
