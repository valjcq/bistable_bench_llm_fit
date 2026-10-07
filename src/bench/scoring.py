"""Three different ways to score a parameter set. They are kept apart on purpose.

    naive_objective   what a user without bifurcation knowledge would write: simulate from a
                      low and a high start and compare where the simulations end up. Cheap,
                      but it can be fooled (see tests/test_scoring.py).
    expert_objective  built on the fixed-point checker: knows about stability and stays
                      graded when the system is monostable. Meant to be minimized.
    evaluation_metric success flag + per-target errors from the checker. Used ONLY for final
                      reporting; it is flat (success or not) so it must never be optimized.

These are internal helpers of the Environment. Optimizers and the agent only ever receive the
Environment's budgeted methods, never these functions or the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from bench.checker import CheckResult
from bench.models.rate_network import RateNetwork
from bench.simulate import SimConfig, simulate_batch


def naive_objective(
    model: RateNetwork,
    params: Mapping[str, float],
    targets: np.ndarray,
    ic_fractions: tuple[float, ...],
    sim_cfg: SimConfig,
    T: float,
) -> float:
    """Mean over targets of the error between a simulated end state and that target.

    Target i is paired with one simulation started with every population at
    ic_fractions[i] * r_max (e.g. one start "low", one "high"). ORDERING CONVENTION: tasks sort
    their targets by increasing mean observed rate (see tasks.make_task), so target 0 is the
    lowest state and ic_fractions should increase too; with unsorted targets the pairing would
    be arbitrary. The error of a pair is the
    max abs difference over the observed populations (the same metric as the checker), and
    the result is the mean over targets. All simulations run in one vectorized call, and the
    whole thing counts as ONE evaluation.

    "Naive" because it only looks at where the simulations happen to stop at time T: it does
    not know if that is a fixed point (a slow transient counts), nor whether the target would
    be stable, nor whether other attractors exist.
    """
    if len(ic_fractions) != len(targets):
        raise ValueError("need one initial-condition fraction per target")
    r_max = model.net(params).r_max
    x0s = np.asarray(ic_fractions)[:, None] * r_max[None, :]  # (K, N)
    sims = simulate_batch(model, params, x0s, T, sim_cfg)
    final_observed = np.array([s.final_state for s in sims])[:, model.observed_idx]
    return float(np.abs(final_observed - targets).max(axis=1).mean())


def expert_objective(
    result: CheckResult,
    targets: np.ndarray,
    penalty: float,
    no_fixed_point_distance: float,
) -> float:
    """Checker-based objective, lower is better, graded even when the system is monostable.

    For each target:
      * matched (a distinct stable state within tol): contributes its real error;
      * unmatched: contributes the distance to the nearest fixed point of ANY kind (stable,
        unstable, saddle: it varies smoothly with the parameters, so there is always a
        gradient to follow) plus the constant `penalty`.
    The result is the mean over targets. The constant penalty is what separates "every target
    matched" (value <= tol) from "something is missing" (value >= penalty / K).

    If the search found no fixed point at all (possible only for N>=2, where it is not
    exhaustive), the distance falls back to `no_fixed_point_distance`.
    The distance metric is the max abs difference over observed populations, like the errors.
    """
    all_observed = np.array([fp.observed for fp in result.fixed_points]).reshape(
        len(result.fixed_points), targets.shape[1]
    )
    contributions = []
    for i, target in enumerate(targets):
        if result.target_matched[i]:
            contributions.append(float(result.target_errors[i]))  # real error, not a cost
        else:
            if len(all_observed):
                d = float(np.abs(all_observed - target).max(axis=1).min())
            else:
                d = no_fixed_point_distance
            contributions.append(d + penalty)
    return float(np.mean(contributions))


@dataclass(frozen=True)
class EvaluationMetric:
    """The final-report summary of one parameter set. Not an objective: do not optimize it."""

    success: bool
    target_errors: list[float | None]  # real error of each target's assigned state
    target_matched: list[bool]
    n_stable: int
    n_extra_stable: int


def evaluation_metric(result: CheckResult) -> EvaluationMetric:
    """Reduce a CheckResult to what is reported at the end of a run."""
    return EvaluationMetric(
        success=result.success,
        target_errors=list(result.target_errors),
        target_matched=list(result.target_matched),
        n_stable=result.n_stable,
        n_extra_stable=result.n_extra_stable,
    )
