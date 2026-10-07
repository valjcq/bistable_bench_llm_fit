"""Environment-side adapter: runs an optimizer against an Environment and keeps the run registry.

The optimizers only know unit-cube objectives. This module is the one place that connects them
to the environment: it maps unit points to physical parameters (`env.space.from_unit`), calls
`env.evaluate(params, objective_name)` (one budgeted evaluation each), and fills
`RunResult.best_params`. It also derives run seeds and stores every finished run so that
`get_run_results` can return it later (the tool layer wraps this class).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from bench.env import OBJECTIVES, Environment
from bench.optimizers import OPTIMIZERS, OptimizerSpec, RunResult


def derive_seed(episode_seed: int, run_counter: int) -> int:
    """Seed of run number `run_counter` (1-based): independent of other runs, reproducible.

    SeedSequence mixes the episode seed and the counter into well-separated streams, so run 2
    does not start where run 1 ended and adding runs never changes earlier ones.
    """
    ss = np.random.SeedSequence(episode_seed, spawn_key=(run_counter,))
    return int(ss.generate_state(1)[0])


class RunRegistry:
    """Launches optimizer runs on one Environment and remembers their results."""

    def __init__(self, env: Environment, episode_seed: int) -> None:
        self._env = env
        self._episode_seed = episode_seed
        self._runs: dict[str, RunResult] = {}

    def specs(self) -> list[OptimizerSpec]:
        return [o.spec for o in OPTIMIZERS.values()]

    def run(
        self,
        optimizer: str,
        objective: str,
        run_budget: int,
        hyperparameters: Mapping[str, Any] | None = None,
        init: Mapping[str, float] | None = None,
        seed: int | None = None,
        stop_on_converged: bool = False,
    ) -> RunResult:
        """Run one optimizer for at most `run_budget` evaluations of `objective`.

        All argument errors raise ValueError before anything is evaluated, so they are free.
        `init` is a physical-units parameter dict; `seed` defaults to a derived one.
        """
        env = self._env
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {optimizer!r}; available: {sorted(OPTIMIZERS)}")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if isinstance(run_budget, bool) or int(run_budget) != run_budget or run_budget < 1:
            raise ValueError("run_budget must be a positive integer")
        if run_budget > env.budget_remaining:
            raise ValueError(
                f"run_budget {run_budget} exceeds the remaining budget {env.budget_remaining}"
            )
        opt = OPTIMIZERS[optimizer]
        opt.spec.resolve(hyperparameters)  # validate now so a bad value fails before the run
        init_unit = None if init is None else np.array(env.space.to_unit(init))
        counter = len(self._runs) + 1
        run_id = f"run{counter}"
        seed = derive_seed(self._episode_seed, counter) if seed is None else int(seed)

        space = env.space

        def objective_fn(x: np.ndarray) -> float:
            return env.evaluate(space.from_unit(x), objective)

        result = opt.run(
            objective_fn,
            d=space.dim,
            run_budget=int(run_budget),
            seed=seed,
            hyperparameters=hyperparameters,
            init=init_unit,
            run_id=run_id,
            stop_on_converged=stop_on_converged,
        )
        if result.best_x is not None:
            result = replace(result, best_params=space.from_unit(result.best_x))
        self._runs[run_id] = result
        return result

    def get(self, run_id: str) -> RunResult:
        if run_id not in self._runs:
            raise ValueError(f"unknown run_id {run_id!r}; known: {sorted(self._runs)}")
        return self._runs[run_id]
