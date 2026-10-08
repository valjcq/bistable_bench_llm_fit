"""The Environment: the ONLY door to the model.

Optimizers and the agent will receive an Environment and nothing else. They can propose
parameters and ask for (a) an objective value, (b) a checker report, (c) a simulation; every such
call costs one evaluation of a fixed budget. The model, the budget counter and the hidden planted
parameters are private and never returned.

Nothing in this package imports anything agent- or LLM-related.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from bench import scoring
from bench.budget import Budget, BudgetExhausted
from bench.checker import CheckResult, check
from bench.config import ExperimentConfig
from bench.models.base import ParamSpace
from bench.models.instances import public_label
from bench.scoring import EvaluationMetric
from bench.simulate import SimResult, simulate_batch, t_max
from bench.tasks import Task

EventHook = Callable[[dict[str, Any]], None]

OBJECTIVES = ("naive", "expert_v2")

# The interface an optimizer may be given: exactly these budgeted methods (plus the read-only
# information properties). `final_report` is deliberately NOT here: it is unbudgeted and flat,
# so it is for the harness only (later, only an agent's submit() will trigger it).
BUDGETED_METHODS = ("evaluate", "check", "simulate")


class Environment:
    """A model + K targets + a budget, behind three budgeted methods.

    Public surface:
      budgeted (1 evaluation each): evaluate, check, simulate   (= BUDGETED_METHODS)
      read-only information:        targets, space, describe(), n_evaluations,
                                    budget_total, budget_remaining, config_hash, rng
      reporting (NOT budgeted):     final_report  (for the harness, see its docstring)

    `event_hook`, if given, is called with a dict after every call (including a refused one),
    so a logger can be attached later. The hook only observes: its return value is ignored.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        targets: np.ndarray,
        seed: int,
        event_hook: EventHook | None = None,
        inputs: Mapping[str, float] | None = None,
    ) -> None:
        self._config = config
        self._model = config.model()
        # External inputs of the task (e.g. the Wilson-Cowan P), told to the agent by describe().
        self._inputs = dict(inputs or {})
        if self._model.task_inputs or self._inputs:
            self._model = self._model.with_inputs(self._inputs)
        targets = np.array(targets, dtype=float)  # private copy
        if targets.shape != (config.task.K, len(self._model.observed)):
            raise ValueError(
                f"targets must have shape ({config.task.K}, {len(self._model.observed)}), "
                f"got {targets.shape}"
            )
        self._targets = targets
        self._budget = Budget(config.budget.max_evaluations)
        self._rng = np.random.default_rng(seed)
        self._hook = event_hook
        self.config_hash = config.hash()  # hash of the fully resolved config

    @classmethod
    def from_task(
        cls, config: ExperimentConfig, task: Task, event_hook: EventHook | None = None
    ) -> Environment:
        """Build the environment for a saved task. Only `task.targets` and `task.seed` are used.

        The task's planted params are never read here, so they cannot leak through the
        environment. Refuses a task that was generated under different model / tolerance /
        checker settings than `config` (its targets would mean something else).
        """
        if task.generation_hash != config.generation_hash():
            raise ValueError(
                "task was generated with different model/task/checker settings than this config"
            )
        if task.K != config.task.K:
            raise ValueError(f"task has K={task.K} but config.task.K={config.task.K}")
        return cls(config, task.targets, task.seed, event_hook, task.inputs)

    # ---- read-only information -------------------------------------------------------

    @property
    def targets(self) -> np.ndarray:
        """The K target states, shape (K, n_observed). A copy: editing it changes nothing."""
        return self._targets.copy()

    @property
    def space(self) -> ParamSpace:
        """Free parameters with bounds and log flags; to_unit / from_unit live here."""
        return self._model.space

    @property
    def rng(self) -> np.random.Generator:
        """Seeded random stream for stochastic components (e.g. an optimizer's restarts)."""
        return self._rng

    @property
    def n_evaluations(self) -> int:
        return self._budget.used

    @property
    def budget_total(self) -> int:
        return self._budget.total

    @property
    def budget_remaining(self) -> int:
        return self._budget.remaining

    def describe(self) -> dict[str, Any]:
        """What an agent may know about the problem (no dynamics, no hidden parameters)."""
        m = self._model
        extra = {"inputs": dict(self._inputs)} if self._inputs else {}
        return {
            **extra,
            "model": public_label(m.name),  # neutral: never the internal name
            "populations": list(m.populations),
            "types": list(m.types),
            "observed": list(m.observed),
            "K": self._config.task.K,
            "tol": self._config.task.tol,
            "targets": self._targets.tolist(),
            "free_parameters": [
                {"name": p.name, "low": p.low, "high": p.high, "log": p.log} for p in m.space.free
            ],
            "budget": self._budget.total,
            "config_hash": self.config_hash,
        }

    # ---- budgeted methods ------------------------------------------------------------

    def evaluate(self, params: Mapping[str, float], objective_name: str) -> float:
        """Objective value of `params` (lower is better). One evaluation.

        objective_name is "naive" (simulate from low/high starts and compare) or "expert_v2"
        (is each target a stable fixed point? graded, no checker call). The final success metric
        is deliberately NOT available here.
        """
        if objective_name not in OBJECTIVES:
            raise ValueError(f"objective_name must be one of {OBJECTIVES}, got {objective_name!r}")
        index = self._begin("evaluate", params)
        cfg = self._config
        if objective_name == "naive":
            value = scoring.naive_objective(
                self._model,
                params,
                self._targets,
                cfg.scoring.naive_ic_fractions,
                cfg.simulator.sim_config(),
                cfg.simulator.T,
            )
        else:
            value = scoring.expert_v2_objective(
                self._model, params, self._targets, cfg.scoring.v2_delta, cfg.scoring.v2_lambda
            )
        self._emit("evaluate", index, params, objective=objective_name, value=value)
        return value

    def check(self, params: Mapping[str, float]) -> CheckResult:
        """Full checker report (fixed points, stability, matching, margins). One evaluation."""
        index = self._begin("check", params)
        result = check(self._model, params, self._targets, self._config.checker)
        self._emit("check", index, params, result=result.to_dict())
        return result

    def simulate(self, params: Mapping[str, float], x0: np.ndarray, T: float) -> SimResult:
        """Simulate one trajectory from `x0` (ALL populations) for time T. One evaluation."""
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (self._model.n_pops,) or not T > 0:
            raise ValueError(f"need x0 of shape ({self._model.n_pops},) and T > 0 (free)")
        self._validate(params)
        cap = t_max(self._config.simulator.sim_config(), self._model.net(params))
        if T > cap:  # checked before charging: a refused horizon costs nothing
            raise ValueError(f"T={T} exceeds the simulation time cap T_max={cap} (free)")
        index = self._begin("simulate", params)
        result = simulate_batch(
            self._model, params, x0[None, :], T, self._config.simulator.sim_config()
        )[0]
        self._emit("simulate", index, params, x0=x0.tolist(), T=T, result=asdict(result))
        return result

    # ---- reporting -------------------------------------------------------------------

    def final_report(self, params: Mapping[str, float]) -> EvaluationMetric:
        """Success flag and per-target errors for the FINAL parameters. NOT budgeted.

        For the harness that scores a finished run, not for the agent: it must not be handed
        to an optimizer (it is flat, and unbudgeted). Hence it is separate from `evaluate`.
        """
        self._validate(params)
        result = check(self._model, params, self._targets, self._config.checker)
        metric = scoring.evaluation_metric(result)
        self._emit("final_report", None, params, metric=asdict(metric))
        return metric

    # ---- internals -------------------------------------------------------------------

    def _validate(self, params: Mapping[str, float]) -> None:
        """Reject unknown/missing names and out-of-bounds values (raises ValueError, free)."""
        self._model.space.to_unit(params)

    def _begin(self, kind: str, params: Mapping[str, float]) -> int:
        """Validate, then charge one evaluation; a refused call is reported to the hook."""
        self._validate(params)
        try:
            return self._budget.charge()
        except BudgetExhausted:
            self._emit("budget_exhausted", None, params, attempted=kind)
            raise

    def _emit(
        self, event: str, index: int | None, params: Mapping[str, float], **data: Any
    ) -> None:
        if self._hook is not None:
            self._hook(
                {
                    "event": event,
                    "index": index,
                    "budget_remaining": self._budget.remaining,
                    "params": {k: float(v) for k, v in params.items()},
                    **data,
                }
            )
