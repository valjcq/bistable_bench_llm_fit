"""CMA-ES through Optuna (needs the `cmaes` package)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler

from bench.optimizers.base import Evaluator, HyperSpec, Optimizer, OptimizerSpec
from bench.optimizers.tpe import ask_tell, enqueue_point


class CMA(Optimizer):
    spec = OptimizerSpec(
        name="cma",
        description=(
            "CMA-ES (Optuna): keeps a Gaussian cloud of candidates and adapts its centre, size "
            "and shape toward better points. Strong local refinement on smooth landscapes; can "
            "stall on flat regions. Starts from `init` if given, else from the box centre."
        ),
        hyperparameters=(
            HyperSpec(
                "sigma0",
                "float",
                0.25,
                "initial step size, as a fraction of the box width",
                low=0.01,
                high=0.5,
            ),
            HyperSpec(
                "popsize",
                "int",
                None,
                "candidates per generation (null = library default 4 + 3 ln d)",
                low=2,
                high=200,
                nullable=True,
            ),
        ),
    )

    def _search(
        self, evaluate: Evaluator, d: int, seed: int, hp: dict[str, Any], init: np.ndarray | None
    ) -> None:
        x0 = np.full(d, 0.5) if init is None else init
        # Optuna 5 deprecates x0/sigma0 (removal in v6) but still supports them; we pin 5.0.0,
        # so silence the warning here instead of letting it flood every run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sampler = CmaEsSampler(
                x0={f"x{i}": float(v) for i, v in enumerate(x0)},
                sigma0=hp["sigma0"],
                popsize=hp["popsize"],
                seed=seed,
                warn_independent_sampling=False,
            )
        study = optuna.create_study(sampler=sampler, direction="minimize")
        # Optuna samples its first trial independently (no search space known yet), so x0 would
        # never be evaluated. Enqueue it: trial 0 is x0, then CMA starts a cloud centred on x0.
        enqueue_point(study, x0)
        ask_tell(study, evaluate, d)
