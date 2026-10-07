"""Random search: the no-intelligence baseline (condition A)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import qmc

from bench.optimizers.base import Evaluator, HyperSpec, Optimizer, OptimizerSpec


class RandomSearch(Optimizer):
    """Evaluates points drawn independently of every score seen so far.

    "uniform" draws iid points. "lhs" draws one Latin hypercube of the whole run budget, which
    covers every axis evenly (no clumps); it needs the run budget up front, which is known.
    """

    spec = OptimizerSpec(
        name="random",
        description=(
            "Random search: evaluates points chosen without looking at earlier scores. "
            "No hyperparameter tuning needed; the baseline any optimizer should beat."
        ),
        hyperparameters=(
            HyperSpec(
                "sampling",
                "choice",
                "uniform",
                "'uniform' = independent uniform points; 'lhs' = Latin hypercube (even coverage)",
                choices=("uniform", "lhs"),
            ),
        ),
    )

    def _search(
        self, evaluate: Evaluator, d: int, seed: int, hp: dict[str, Any], init: np.ndarray | None
    ) -> None:
        rng = np.random.default_rng(seed)
        if init is not None:
            evaluate(init)  # the starting point is evaluated first and counts as one eval
        n = evaluate.run_budget - len(evaluate.history)
        if hp["sampling"] == "lhs":
            points = qmc.LatinHypercube(d=d, rng=rng).random(n)
            for x in points:
                evaluate(x)
        else:
            while True:  # Evaluator stops us at the run budget
                evaluate(rng.random(d))
