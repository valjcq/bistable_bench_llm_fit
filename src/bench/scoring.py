"""Three different ways to score a parameter set. They are kept apart on purpose.

    naive_objective   what a user without bifurcation knowledge would write: simulate from a
                      low and a high start and compare where the simulations end up. Cheap,
                      but it can be fooled (see tests/test_scoring.py).
    expert_v2_objective  candidate replacement of v1 (no checker call): asks of each target
                      whether it is a stable fixed point. See NOTES.md for the definition.
    expert_v1_objective  built on the fixed-point checker: knows about stability and stays
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
from scipy.optimize import brentq

from bench.checker import CheckResult
from bench.models.rate_network import RateNetwork, fixed_point_residual, jacobian
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


def expert_v1_objective(
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


def expert_v2_terms(
    model: RateNetwork, params: Mapping[str, float], targets: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-target (residual_k, stab_k) of the expert_v2 objective (definition: NOTES.md).

    Each target is completed to a full state r*: observed populations take the target values,
    and the one unobserved population (if any) solves its own fixed-point equation
    -x + F_i(W_ii x + sum_{j != i} W_ij r_j + I_i) = 0 given the others. The left side is >= 0 at
    x = 0 and <= 0 at x = r_max (F lies in [0, r_max]), so a root is bracketed and `brentq`
    finds it; for the E/I model's inhibitory population it is also unique (decreasing in x).

    residual_k says how far r* is from being a fixed point (zero iff the target IS one);
    stab_k says how unstable it is: max Re eig(J) times the slowest tau is a dimensionless decay
    rate, `delta` demands a small margin of stability before the penalty reaches zero.
    """
    net = model.net(params)
    obs = model.observed_idx
    unobs = [i for i in range(model.n_pops) if i not in obs]
    if len(unobs) > 1:
        raise ValueError("expert_v2 supports at most one unobserved population")
    tau_max = float(net.tau.max())
    residuals, stabs = [], []
    for target in targets:
        r = np.zeros(model.n_pops)
        r[obs] = target
        if unobs:
            i = unobs[0]

            def g(x: float, i: int = i, r: np.ndarray = r) -> float:
                r2 = r.copy()
                r2[i] = x
                return float(fixed_point_residual(r2, net)[i])

            hi = float(net.r_max[i])
            g_lo, g_hi = g(0.0), g(hi)
            if g_lo <= 0.0:
                r[i] = 0.0
            elif g_hi >= 0.0:
                r[i] = hi
            else:
                r[i] = brentq(g, 0.0, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
        residuals.append(float(np.abs(fixed_point_residual(r, net)[obs]).max()))
        max_re = float(np.linalg.eigvals(jacobian(r, net)).real.max())
        stabs.append(max(0.0, max_re * tau_max + delta))
    return np.array(residuals), np.array(stabs)


def expert_v2_objective(
    model: RateNetwork,
    params: Mapping[str, float],
    targets: np.ndarray,
    delta: float,
    lam: float,
) -> float:
    """expert_v2: mean over targets of (residual_k + lam * stab_k); lower is better.

    It asks of each target directly "is this state a stable fixed point of these parameters?"
    instead of searching for fixed points and matching them, so it needs no checker call, has no
    tolerance-dependent jump and no flat plateau where a state sits on only one target. It is
    zero exactly when every target is a fixed point that is stable with margin `delta`.
    Success is NOT decided here: only the checker decides.
    """
    residuals, stabs = expert_v2_terms(model, params, targets, delta)
    return float(np.mean(residuals + lam * stabs))
