"""Helpers for the Wilson-Cowan (1972) model: the paper's parameter sets and a P-sweep model.

The model itself is the `wc1972` builder (instances.py) with the YAML in configs/wc1972.yaml.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from bench.models.base import ParamSpace, ParamSpec
from bench.models.rate_network import RateNetwork

PAPER_SETS_PATH = Path(__file__).resolve().parents[3] / "configs/wc1972_paper_sets.yaml"
FREE = ("c1", "c2", "c3", "c4", "a_e", "theta_e", "a_i", "theta_i")


def paper_sets(path: str | Path = PAPER_SETS_PATH) -> dict[str, dict[str, float]]:
    """The card's parameter sets: name -> {c1..theta_i (the free parameters), 'P'}."""
    raw = yaml.safe_load(Path(path).read_text())
    return {name: {k: float(v) for k, v in s.items()} for name, s in raw.items()}


def free_params(paper_set: dict[str, float]) -> dict[str, float]:
    """The eight free parameters of a paper set (without its input P)."""
    return {k: paper_set[k] for k in FREE}


def with_free_input(model: RateNetwork, low: float = -2.0, high: float = 2.0) -> RateNetwork:
    """The same model with the input P moved from the fixed to the free parameters.

    Lets code sweep P (the paper's hysteresis figures) through the ordinary
    `model.net(params)` interface. The default config keeps P fixed at 0.
    """
    fixed = {k: v for k, v in model.space.fixed.items() if k != "P"}
    space = ParamSpace((*model.space.free, ParamSpec("P", low, high)), fixed)
    return replace(model, space=space)
