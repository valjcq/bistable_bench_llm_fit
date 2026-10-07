"""Common machinery of the optimizer wrappers.

Optimizers live on the unit cube [0, 1]^d and know nothing about the model, the checker or the
environment: they get an objective callable and ask it for values, nothing else. Everything the
three optimizers have in common (counting evaluations, the run budget, the global-budget stop,
the best-so-far record, the convergence flag) is done once here, in `Optimizer.run`; a concrete
optimizer only implements `_search`, the loop that proposes points.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bench.budget import BudgetExhausted  # just an exception class: no model/env code

Objective = Callable[[np.ndarray], float]  # unit-cube point (d,) -> score, lower is better

CONVERGENCE_EPS = 1e-6  # "no real improvement" threshold of the convergence flag


@dataclass(frozen=True)
class HyperSpec:
    """One tunable hyperparameter of an optimizer, as shown to the agent by list_optimizers."""

    name: str
    type: str  # "float" | "int" | "bool" | "choice"
    default: Any
    description: str
    low: float | None = None  # range of a float/int
    high: float | None = None
    choices: tuple[str, ...] | None = None  # allowed values of a "choice"
    nullable: bool = False  # None is allowed (means "use the library default")


@dataclass(frozen=True)
class OptimizerSpec:
    """Name, purpose and hyperparameters of an optimizer (becomes one list_optimizers entry)."""

    name: str
    description: str
    hyperparameters: tuple[HyperSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)  # tuples become lists, so the result is plain JSON
        d["hyperparameters"] = [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()}
            for h in d["hyperparameters"]
        ]
        return d

    def resolve(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate `given` against the spec and fill in defaults. Raises ValueError (free)."""
        given = dict(given or {})
        known = {h.name: h for h in self.hyperparameters}
        unknown = set(given) - set(known)
        if unknown:
            raise ValueError(
                f"unknown hyperparameter(s) {sorted(unknown)} for '{self.name}'; "
                f"allowed: {sorted(known)}"
            )
        out = {}
        for name, h in known.items():
            out[name] = _check_value(h, given.get(name, h.default), self.name)
        return out


def _check_value(h: HyperSpec, v: Any, opt: str) -> Any:
    """Validate one hyperparameter value against its spec; return it (ints stay int)."""
    where = f"hyperparameter '{h.name}' of '{opt}'"
    if v is None:
        if h.nullable:
            return None
        raise ValueError(f"{where} must not be null")
    if h.type == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{where} must be a boolean, got {v!r}")
    elif h.type == "choice":
        if v not in (h.choices or ()):
            raise ValueError(f"{where} must be one of {list(h.choices or ())}, got {v!r}")
    elif h.type in ("int", "float"):
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise ValueError(f"{where} must be a number, got {v!r}")
        if h.type == "int":
            if int(v) != v:
                raise ValueError(f"{where} must be an integer, got {v!r}")
            v = int(v)
        else:
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"{where} must be finite")
        if (h.low is not None and v < h.low) or (h.high is not None and v > h.high):
            raise ValueError(f"{where} must be in [{h.low}, {h.high}], got {v!r}")
    return v


@dataclass(frozen=True)
class RunResult:
    """Everything stored about one optimizer run. JSON-serializable via to_dict."""

    run_id: str
    optimizer: str
    hyperparameters: dict[str, Any]
    seed: int
    evals_used: int
    best_score: float | None  # None only if not a single evaluation succeeded
    best_x: list[float] | None  # best point in unit space
    best_params: dict[str, float] | None  # best point in physical units (filled by the adapter)
    history: list[float]  # score of every evaluation, in call order
    converged: bool
    stopped_reason: str  # "run_budget" | "global_budget" | "converged"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_converged(history: list[float], run_budget: int) -> bool:
    """The convergence flag, identical for every optimizer.

    Converged means the best-so-far score improved by less than 1e-6 over the last
    w = max(20, 20% of the run budget) evaluations. It needs more than w evaluations so that the
    window is fully inside the run. It is a stagnation signal, not a proof of optimality.
    """
    w = max(20, math.ceil(0.2 * run_budget))
    if len(history) <= w:
        return False
    best = np.minimum.accumulate(np.asarray(history, dtype=float))
    return bool(best[-w - 1] - best[-1] < CONVERGENCE_EPS)


class _StopRun(Exception):
    """Internal: tells `_search` to unwind. Carries the stopped_reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Evaluator:
    """Wraps the objective for one run: counts calls and records the history and the best point.

    `_search` implementations call it like the objective. It raises _StopRun when the run
    budget is spent, when the global budget runs out underneath the objective, or (optionally)
    when the run has converged, so no optimizer needs its own bookkeeping.
    """

    def __init__(self, f: Objective, run_budget: int, stop_on_converged: bool) -> None:
        self._f = f
        self.run_budget = run_budget
        self._stop_on_converged = stop_on_converged
        self.history: list[float] = []
        self.best_score = math.inf
        self.best_x: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> float:
        if len(self.history) >= self.run_budget:
            raise _StopRun("run_budget")
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)  # the cube is a hard box
        try:
            score = float(self._f(x))
        except BudgetExhausted:
            raise _StopRun("global_budget") from None
        self.history.append(score)
        if score < self.best_score:  # strict: on ties the earliest point stays the best
            self.best_score, self.best_x = score, x.copy()
        if self._stop_on_converged and is_converged(self.history, self.run_budget):
            raise _StopRun("converged")
        return score


class Optimizer(ABC):
    """Base class: subclasses set `spec` and implement `_search`."""

    spec: OptimizerSpec

    @abstractmethod
    def _search(
        self,
        evaluate: Evaluator,
        d: int,
        seed: int,
        hp: dict[str, Any],
        init: np.ndarray | None,
    ) -> None:
        """Propose points and call `evaluate(x)` on each until it raises _StopRun.

        `init`, if given, is a unit-cube point that must be evaluated first (as the starting
        point of the search).
        """

    def run(
        self,
        objective: Objective,
        d: int,
        run_budget: int,
        seed: int,
        hyperparameters: Mapping[str, Any] | None = None,
        init: np.ndarray | None = None,
        run_id: str = "",
        stop_on_converged: bool = False,
    ) -> RunResult:
        """Run the optimizer for at most `run_budget` evaluations and report what happened.

        `stop_on_converged` ends the run early when the convergence flag is raised; baselines
        leave it off so that every condition spends its whole budget.
        """
        if d < 1 or run_budget < 1:
            raise ValueError("d and run_budget must be >= 1")
        hp = self.spec.resolve(hyperparameters)
        if init is not None:
            init = np.asarray(init, dtype=float)
            if init.shape != (d,) or np.any(init < 0) or np.any(init > 1):
                raise ValueError(f"init must have shape ({d},) with values in [0, 1]")
        ev = Evaluator(objective, run_budget, stop_on_converged)
        reason = "run_budget"  # also what happens if _search returns without being stopped
        try:
            self._search(ev, d, int(seed), hp, init)
        except _StopRun as stop:
            reason = stop.reason
        return RunResult(
            run_id=run_id,
            optimizer=self.spec.name,
            hyperparameters=hp,
            seed=int(seed),
            evals_used=len(ev.history),
            best_score=None if ev.best_x is None else ev.best_score,
            best_x=None if ev.best_x is None else [float(v) for v in ev.best_x],
            best_params=None,
            history=list(ev.history),
            converged=is_converged(ev.history, run_budget),
            stopped_reason=reason,
        )
