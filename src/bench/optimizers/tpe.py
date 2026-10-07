"""TPE (Tree-structured Parzen Estimator) through Optuna."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from bench.optimizers.base import Evaluator, HyperSpec, Optimizer, OptimizerSpec

optuna.logging.set_verbosity(optuna.logging.CRITICAL)  # no per-trial chatter in logs


def ask_tell(study: optuna.Study, evaluate: Evaluator, d: int) -> None:
    """Optuna loop shared by TPE and CMA-ES: ask a point, evaluate it, tell the score back.

    ask/tell (instead of study.optimize) lets our Evaluator own the budget and the stopping.
    Non-finite scores are replaced by a large finite number because Optuna rejects NaN.
    """
    while True:
        trial = study.ask()
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(d)])
        score = evaluate(x)  # raises _StopRun at the end; the pending trial is just dropped
        study.tell(trial, score if np.isfinite(score) else 1e9)


def enqueue_point(study: optuna.Study, x: np.ndarray) -> None:
    """Force Optuna's next trial to be exactly the unit-cube point x."""
    study.enqueue_trial({f"x{i}": float(v) for i, v in enumerate(x)})


class TPE(Optimizer):
    spec = OptimizerSpec(
        name="tpe",
        description=(
            "Tree-structured Parzen Estimator (Optuna). After a random start-up phase it models "
            "good and bad points separately and samples where good ones are likely. Handles "
            "rough, noisy landscapes; slower to converge precisely than CMA-ES."
        ),
        hyperparameters=(
            HyperSpec(
                "n_startup_trials",
                "int",
                10,
                "number of random evaluations before the model is used",
                low=1,
                high=1000,
            ),
            HyperSpec(
                "multivariate",
                "bool",
                True,
                "model parameters jointly (captures interactions) instead of one at a time",
            ),
        ),
    )

    def _search(
        self, evaluate: Evaluator, d: int, seed: int, hp: dict[str, Any], init: np.ndarray | None
    ) -> None:
        sampler = TPESampler(
            seed=seed,
            n_startup_trials=hp["n_startup_trials"],
            multivariate=hp["multivariate"],
        )
        study = optuna.create_study(sampler=sampler, direction="minimize")
        if init is not None:
            enqueue_point(study, init)  # the first trial is exactly init
        ask_tell(study, evaluate, d)
