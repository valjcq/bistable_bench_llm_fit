"""Experiment configuration: YAML -> typed, validated dataclasses, plus stable hashes.

An experiment file has these sections (all but `model` optional, defaults below):

    model:     path of a model YAML (relative to the experiment file)
    task:      K, tol, noise, seed, ... (how targets are planted and what counts as success)
    checker:   numerical settings of the fixed-point checker
    simulator: RK4 settings and the default time horizon T
    scoring:   settings of the naive / expert objectives
    budget:    number of allowed evaluations
    baseline:  which optimizer conditions B and C use

Every section is checked strictly: an unknown key (usually a typo) is an error, so a config
can never silently run with a setting that was meant to be something else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from bench.checker import CheckerConfig
from bench.models.instances import model_from_dict
from bench.models.rate_network import RateNetwork
from bench.simulate import SimConfig


@dataclass(frozen=True)
class TaskConfig:
    """How a task is defined (K targets, tolerance) and planted."""

    K: int = 2  # number of target states
    tol: float = 0.02  # success: abs error per observed population <= tol
    noise: float = 0.0  # std (in rate units) of Gaussian noise added to the planted targets
    seed: int = 0  # default seed for task generation / environment randomness
    forbid_extra_stable_states: bool = False  # if True, extra stable states make success False
    # Planted target rates must lie inside this window, as fractions of r_max: a target near
    # 0 or r_max sits in the saturated tail where a tolerance of `tol` is meaningless.
    target_window: tuple[float, float] = (0.05, 0.95)
    # Planted targets must be farther apart than factor * tol (max abs difference over observed
    # populations), so their tolerance boxes are clearly distinct.
    min_separation_tol_factor: float = 3.0
    max_attempts: int = 5000  # random parameter draws before giving up
    prefilter_starts: int = 16  # multi-start count of the cheap prefilter (N>=2)

    def __post_init__(self) -> None:
        lo, hi = self.target_window
        if self.K < 2:
            raise ValueError("task.K must be >= 2")
        if self.tol <= 0 or self.noise < 0 or self.min_separation_tol_factor < 0:
            raise ValueError("task: tol must be > 0, noise and min_separation_tol_factor >= 0")
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("task.target_window must satisfy 0 <= lo < hi <= 1")
        if self.max_attempts < 1 or self.prefilter_starts < 1:
            raise ValueError("task: max_attempts and prefilter_starts must be >= 1")


@dataclass(frozen=True)
class BudgetConfig:
    max_evaluations: int = 200

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ValueError("budget.max_evaluations must be >= 1")


@dataclass(frozen=True)
class SimulatorConfig:
    """RK4 settings (see SimConfig) plus the time horizon T used by the objectives."""

    dt: float = 0.1
    flow_tol: float = 1.0e-6
    hold_tau_factor: float = 5.0
    T: float = 500.0  # max simulated time; several times the slowest tau
    T_max: float | None = None  # cap on any Environment.simulate horizon; None = 200 * max(tau)

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("simulator.T must be > 0")
        self.sim_config()  # validates dt, flow_tol, hold_tau_factor

    def sim_config(self) -> SimConfig:
        return SimConfig(self.dt, self.flow_tol, self.hold_tau_factor, self.T_max)


@dataclass(frozen=True)
class ScoringConfig:
    # Naive objective: target i is compared with the end state of a simulation started with
    # EVERY population at naive_ic_fractions[i] * r_max. One entry per target (K entries);
    # targets are sorted by increasing rate, so a low IC goes with the low target.
    naive_ic_fractions: tuple[float, ...] = (0.0, 1.0)
    # Expert objective (definition in NOTES.md): stability margin demanded of every target state, in
    # units of the slowest tau, and the weight of the instability term against the residual.
    v2_delta: float = 0.01
    v2_lambda: float = 1.0

    def __post_init__(self) -> None:
        if not self.naive_ic_fractions or any(not 0 <= f <= 1 for f in self.naive_ic_fractions):
            raise ValueError("scoring.naive_ic_fractions must be non-empty, values in [0, 1]")
        if self.v2_delta < 0 or self.v2_lambda <= 0:
            raise ValueError("scoring.v2_delta must be >= 0 and scoring.v2_lambda > 0")


@dataclass(frozen=True)
class BaselineConfig:
    """Settings of the scripted baselines (conditions A-C)."""

    optimizer: str = "cma"  # the optimizer of conditions B and C: "cma" or "tpe"
    # Submission rule (the same for every baseline): this many evaluations are reserved from the
    # run's own budget; at the end the top-k distinct candidates by the condition's objective
    # are checked with the checker, and the best-scored one that passes is submitted (else the
    # best-scored one). 0 = no reserve, submit the best-scored point (the Phase 0 rule).
    submission_checks: int = 3

    def __post_init__(self) -> None:
        if self.optimizer not in ("cma", "tpe"):
            raise ValueError("baseline.optimizer must be 'cma' or 'tpe'")
        if self.submission_checks < 0:
            raise ValueError("baseline.submission_checks must be >= 0")


def _section(cls: type, raw: dict[str, Any] | None, name: str, **extra: Any) -> Any:
    """Build dataclass `cls` from a raw YAML dict, rejecting unknown keys and lists->tuples."""
    raw = dict(raw or {})
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) in '{name}': {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    clash = set(raw) & set(extra)
    if clash:
        raise ValueError(f"'{name}' must not set {sorted(clash)} (it comes from another section)")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**kwargs, **extra)


@dataclass(frozen=True)
class ExperimentConfig:
    """A fully resolved experiment: the model spec is inlined, every section is typed."""

    model_spec: dict[str, Any]  # the parsed model YAML (inlined, so hashes cover its content)
    task: TaskConfig = field(default_factory=TaskConfig)
    checker: CheckerConfig = field(default_factory=CheckerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def __post_init__(self) -> None:
        self.model()  # validates the model spec (raises on a bad one)
        if (self.checker.tol, self.checker.forbid_extra_stable_states) != (
            self.task.tol,
            self.task.forbid_extra_stable_states,
        ):
            raise ValueError("checker.tol / forbid_extra_stable_states must come from `task`")
        if len(self.scoring.naive_ic_fractions) != self.task.K:
            raise ValueError(
                f"scoring.naive_ic_fractions needs one entry per target (K={self.task.K}), "
                f"got {len(self.scoring.naive_ic_fractions)}"
            )

    def model(self) -> RateNetwork:
        """Build the model object described by `model_spec`."""
        return model_from_dict(self.model_spec)

    def resolved(self) -> dict[str, Any]:
        """Everything that defines the experiment, defaults included, as plain JSON types."""
        d = {
            "model": self.model_spec,
            "task": asdict(self.task),
            "checker": asdict(self.checker),
            "simulator": asdict(self.simulator),
            "scoring": asdict(self.scoring),
            "budget": asdict(self.budget),
            "baseline": asdict(self.baseline),
        }
        return json.loads(json.dumps(d))  # tuples -> lists, so the form is canonical

    def hash(self) -> str:
        """Stable hash of the full resolved config (stored on the Environment)."""
        return _hash(self.resolved())

    def generation_hash(self) -> str:
        """Hash of only what determines how tasks are planted and judged.

        Excludes the per-task `seed` and `K` (a task file records those itself), `max_attempts`
        (it only decides when generation gives up, never what a generated task looks like) and
        everything about running (budget, simulator, scoring). A task file therefore stays valid
        when, say, the budget changes, but is rejected if the model, tolerance or checker
        numerics changed.
        """
        r = self.resolved()
        task = {k: v for k, v in r["task"].items() if k not in ("seed", "K", "max_attempts")}
        return _hash({"model": r["model"], "task": task, "checker": r["checker"]})


def _hash(obj: Any) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace): identical content, same hash."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def config_from_dict(
    raw: dict[str, Any], model_spec: dict[str, Any], resolved: bool = False
) -> ExperimentConfig:
    """Build an ExperimentConfig from the raw sections of an experiment file.

    With `resolved=True`, `raw` is the output of `ExperimentConfig.resolved()` (that is how a run
    directory's config is turned back into a config); the checker's tol /
    forbid_extra_stable_states keys, which `resolved()` repeats from `task`, are then ignored.
    In a hand-written file they stay an error.
    """
    unknown = set(raw) - {"model", "task", "checker", "simulator", "scoring", "budget", "baseline"}
    if unknown:
        raise ValueError(f"unknown top-level key(s): {sorted(unknown)}")
    task = _section(TaskConfig, raw.get("task"), "task")
    checker_raw = {
        k: v
        for k, v in (raw.get("checker") or {}).items()
        if not (resolved and k in ("tol", "forbid_extra_stable_states"))
    }
    checker = _section(
        CheckerConfig,
        checker_raw,
        "checker",
        tol=task.tol,
        forbid_extra_stable_states=task.forbid_extra_stable_states,
    )
    return ExperimentConfig(
        model_spec=model_spec,
        task=task,
        checker=checker,
        simulator=_section(SimulatorConfig, raw.get("simulator"), "simulator"),
        scoring=_section(ScoringConfig, raw.get("scoring"), "scoring"),
        budget=_section(BudgetConfig, raw.get("budget"), "budget"),
        baseline=_section(BaselineConfig, raw.get("baseline"), "baseline"),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read an experiment YAML (and the model YAML it points to) into an ExperimentConfig."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    model_spec = yaml.safe_load((path.parent / raw["model"]).read_text())
    return config_from_dict({k: v for k, v in raw.items() if k != "model"}, model_spec)
