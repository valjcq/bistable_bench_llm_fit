"""Fixed-point checker: the ground truth for "does this parameter set realize these targets?".

Given a model, parameters and K target states, the checker
  1. finds the fixed points of the network,
  2. classifies each as stable / unstable from the Jacobian eigenvalues,
  3. matches targets to distinct stable states and decides `success`,
  4. reports how robust the match is (margins).

It never simulates. Simulation (simulate.py) is the independent cross-check.

Fixed-point search:
  * N=1: exact (grid scan + brentq), see `find_fixed_points_1d`.
  * N>=2: Newton from the targets (local, cheap) + multi-start from a seeded Latin hypercube
    (global, NOT exhaustive), see `find_fixed_points_nd`.

The search is a pure function of (params, targets, config): the seed lives in the config, not
in a shared random stream, so the same parameters always give the same answer no matter what
was evaluated before. Optimizers rely on that.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import brentq, linear_sum_assignment, minimize_scalar, root
from scipy.stats import qmc

from bench.models.base import NetArrays
from bench.models.rate_network import (
    RateNetwork,
    fixed_point_residual,
    jacobian,
    rate_fn,
    residual_jacobian,
)


@dataclass(frozen=True)
class CheckerConfig:
    """Numerical settings of the checker (all deterministic)."""

    tol: float = 0.02  # max abs error per observed population for a target to count as matched
    eps_stab: float = 1e-6  # stable <=> max Re(eig) < -eps_stab (so marginal points are NOT stable)
    forbid_extra_stable_states: bool = False  # if True, extra stable states make success False
    # --- N=1 exact scan ---
    n_grid_1d: int = 20001  # grid nodes on [0, r_max]
    dedup_tol: float = 1e-9  # 1D roots closer than this are the same fixed point
    near_tangent_tol: float = 1e-4  # |g| below this at a local extremum triggers refinement
    # --- N>=2 search ---
    seed: int = 0  # seeds the Latin hypercube sample of the global search
    n_starts: int = 256  # number of global multi-start Newton runs
    n_unobserved_grid: int = 5  # local search: grid levels per UNobserved population
    residual_tol: float = 1e-10  # a Newton result counts as a root if max|g(r)| is below this
    box_slack: float = 1e-9  # roots must lie in [-slack, r_max + slack]^N
    dedup_tol_nd: float = 1e-6  # N>=2 roots closer than this (Euclidean) are the same

    def __post_init__(self) -> None:
        if self.tol <= 0 or self.eps_stab < 0 or self.n_grid_1d < 5:
            raise ValueError("invalid CheckerConfig")
        if self.n_starts < 1 or self.n_unobserved_grid < 1:
            raise ValueError("invalid CheckerConfig")


@dataclass(frozen=True)
class FixedPoint:
    """One fixed point with its linear-stability information."""

    r: list[float]  # full state (all populations)
    observed: list[float]  # rates of the observed populations only
    eig_real: list[float]  # eigenvalues of the Jacobian (real parts) ...
    eig_imag: list[float]  # ... and imaginary parts (kept separate: JSON has no complex)
    max_re: float  # max real part of the eigenvalues; < -eps_stab means stable
    stable: bool


@dataclass(frozen=True)
class CheckResult:
    """Everything the checker knows. Plain lists/floats/bools only, so it is JSON-serializable.

    Index convention: every state index in this result (`target_assignment`, `stable_indices`,
    `extra_stable_indices`) refers to a position in `fixed_points`.
    """

    fixed_points: list[FixedPoint]  # all fixed points found, sorted by r
    stable_indices: list[int]  # which of `fixed_points` are stable
    n_stable: int
    # Matching, one entry per target (same order as the targets passed in). Every target that
    # could be given a stable state is assigned one and its REAL error is reported, even if that
    # error is above tol (the matching is one-to-one, so this happens when states run out of
    # in-tolerance partners):
    target_assignment: list[int | None]  # index into fixed_points; None = no stable state left
    target_errors: list[float | None]  # max abs error over observed pops; None if unassigned
    target_matched: list[bool]  # assigned AND error <= tol
    extra_stable_indices: list[int]  # stable states not assigned to ANY target
    n_extra_stable: int
    success: bool
    # Robustness of the matched states (None when undefined, e.g. nothing matched):
    margin_eig: float | None  # min over matched states of -max Re(eig): slowest decay rate
    margin_dist: float | None  # min distance from a matched state to the nearest unstable fp
    global_search_exhaustive: bool  # True only when the fixed-point list is provably complete
    warnings: list[str] = field(default_factory=list)  # suspicious-search notes (N>=2)
    tol: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def describe_fixed_point(
    model: RateNetwork, net: NetArrays, r: np.ndarray, eps_stab: float
) -> FixedPoint:
    """Attach linear-stability information to a state `r` that is (numerically) a fixed point.

    Computes the Jacobian at r and its eigenvalues. Near a fixed point the dynamics are
    dx/dt = J x, so r is stable iff every eigenvalue has negative real part. We require
    max Re(eig) < -eps_stab rather than < 0 so that numerically marginal points (a
    saddle-node bifurcation sitting right on the parameters) are not called stable.
    """
    eig = np.linalg.eigvals(jacobian(r, net))
    max_re = float(np.max(eig.real))
    return FixedPoint(
        r=[float(x) for x in r],
        observed=[float(x) for x in r[model.observed_idx]],
        eig_real=[float(x) for x in eig.real],
        eig_imag=[float(x) for x in eig.imag],
        max_re=max_re,
        stable=bool(max_re < -eps_stab),
    )


def find_fixed_points_1d(net: NetArrays, cfg: CheckerConfig) -> list[float]:
    """All fixed points of a 1-population network, found exactly (up to the limit below).

    Why this is exact: F is bounded in (0, r_max), so every fixed point of
    g(r) = -r + F(w r + I) lies in [0, r_max]. We evaluate g on a dense grid there and
    collect roots from three sources:
      (a) grid nodes where g is exactly 0;
      (b) strict sign changes between neighbouring nodes, refined with brentq;
      (c) near-tangent roots, which (b) cannot see: a pair of roots (or a double root) that
          fit between two nodes leaves g with the same sign at both. We detect a local
          extremum of g on the grid whose |g| is tiny, then locate the extremum precisely
          with a bounded minimizer. If g changes sign there, the two sides each bracket a
          root (brentq); if g only touches zero, that is a double root.
    Roots from all sources are then sorted and de-duplicated (closer than cfg.dedup_tol),
    which also makes sure a root sitting exactly on a node is counted once.

    Grid limit and why it is acceptable. With the default grid the spacing is
    h = r_max / 20000 = 5e-5. A pair of roots can only slip through if
      1. both roots lie inside one cell (their separation is < h), and
      2. |g| at the neighbouring nodes is above `near_tangent_tol`, so (c) is not triggered.
    A pair that close means the parameters sit at a saddle-node (fold) within about 1e-5 or
    less (relative): the dip of g below zero inside a cell is at most g'' h^2 / 8, i.e. ~3e-10
    times g''. Condition 2 would additionally need g'' > ~3e5, but within the parameter bounds
    used here g'' <~ 1e3, so (c) always fires first; the limit is purely theoretical for the
    shipped configs. And even if such a pair were missed, the stable state of that pair has
    |lambda| ~ sqrt(distance to the fold): it is marginal, a useless target for fitting.
    A tangent pair closer than `dedup_tol` is reported as one (marginal, hence non-stable) root.
    """
    r_max = float(net.r_max[0])
    w = float(net.W[0, 0])

    def g(r: float) -> float:
        return float(-r + rate_fn(np.array([w * r + net.I_ext[0]]), net)[0])

    grid = np.linspace(0.0, r_max, cfg.n_grid_1d)
    # Same formula as g, vectorized over the grid for speed (rate_fn broadcasts over shape (M,)).
    gv = -grid + rate_fn(w * grid + net.I_ext[0], net)

    roots: list[float] = []
    # (a) exact zeros on nodes
    roots += [float(r) for r in grid[gv == 0.0]]
    # (b) strict sign changes between neighbouring nodes
    for i in np.nonzero(gv[:-1] * gv[1:] < 0)[0]:
        roots.append(brentq(g, grid[i], grid[i + 1], xtol=1e-14))
    # (c) near-tangent roots at local extrema of g with tiny |g|
    d = np.diff(gv)
    extrema = np.nonzero((d[:-1] * d[1:] < 0) & (np.abs(gv[1:-1]) < cfg.near_tangent_tol))[0] + 1
    for i in extrema:
        lo, hi = grid[i - 1], grid[i + 1]
        sign = 1.0 if gv[i] > 0 else -1.0  # local min if g>0, local max if g<0: push toward 0
        res = minimize_scalar(
            lambda r, s=sign: s * g(r), bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-13},
        )  # fmt: skip
        r_ext = float(res.x)
        if sign * g(r_ext) < 0 and g(lo) * g(r_ext) < 0:  # extremum crossed zero: two roots
            roots.append(brentq(g, lo, r_ext, xtol=1e-14))
            roots.append(brentq(g, r_ext, hi, xtol=1e-14))
        elif abs(g(r_ext)) < 1e-12:  # touches zero: double root
            roots.append(r_ext)

    roots.sort()
    unique: list[float] = []
    for r in roots:
        if not unique or r - unique[-1] > cfg.dedup_tol:
            unique.append(r)
    return unique


def _newton(net: NetArrays, start: np.ndarray, cfg: CheckerConfig) -> np.ndarray | None:
    """Run Newton (scipy `root`, method 'hybr') on g(r) = 0 from `start`; None if it fails.

    Uses the analytic Jacobian of g. A result is accepted only if it converged, the residual
    is really tiny, and it lies inside the physical box [0, r_max]^N (F is bounded, so any
    true fixed point does; a root outside would be a numerical artifact).
    """
    res = root(
        lambda r: fixed_point_residual(r, net),
        start,
        jac=lambda r: residual_jacobian(r, net),
        method="hybr",
        tol=1e-12,
    )
    if not res.success:
        return None
    r = res.x
    if np.max(np.abs(fixed_point_residual(r, net))) > cfg.residual_tol:
        return None
    if np.any(r < -cfg.box_slack) or np.any(r > net.r_max + cfg.box_slack):
        return None
    return r


def find_fixed_points_nd(
    model: RateNetwork, net: NetArrays, targets: np.ndarray, cfg: CheckerConfig
) -> list[np.ndarray]:
    """Fixed points of an N>=2 network: a LOCAL search around the targets plus a GLOBAL one.

    NOT exhaustive: there is no cheap way to guarantee all roots of a nonlinear system in
    N dimensions, so the caller reports `global_search_exhaustive = False`.

    (a) LOCAL: start Newton from each target. The target fixes the observed populations; the
        unobserved ones (e.g. the inhibitory rate when only E is observed) are unknown, so we
        start from every point of a small grid over them (cfg.n_unobserved_grid levels each).
        This is the part that matters for success: if the parameters really realize a target,
        a fixed point sits next to it and Newton converges to it quickly and exactly.
    (b) GLOBAL: Newton from cfg.n_starts points of a seeded Latin hypercube over the box
        [0, r_max]^N. It discovers the OTHER fixed points (extra stable states, the saddles
        between states) that margins and the expert objective need. It can miss some.

    Roots from both parts are merged, de-duplicated by Euclidean distance (cfg.dedup_tol_nd)
    and sorted, so the result is deterministic.
    """
    n = model.n_pops
    r_max = net.r_max
    obs = model.observed_idx
    unobs = np.array([i for i in range(n) if i not in set(obs.tolist())], dtype=int)

    starts: list[np.ndarray] = []
    levels = (np.arange(cfg.n_unobserved_grid) + 0.5) / cfg.n_unobserved_grid  # cell midpoints
    for target in targets:
        for combo in itertools.product(levels, repeat=len(unobs)):
            s = np.zeros(n)
            s[obs] = np.clip(target, 0.0, r_max[obs])
            s[unobs] = np.array(combo, dtype=float) * r_max[unobs]
            starts.append(s)
    lhs = qmc.LatinHypercube(d=n, rng=np.random.default_rng(cfg.seed)).random(cfg.n_starts)
    starts += list(lhs * r_max)

    unique: list[np.ndarray] = []
    for s in starts:
        r = _newton(net, s, cfg)
        if r is not None and all(np.linalg.norm(r - u) > cfg.dedup_tol_nd for u in unique):
            unique.append(r)
    unique.sort(key=lambda r: tuple(r))
    return unique


def _match(
    errors: np.ndarray, tol: float
) -> tuple[list[int | None], list[float | None], list[bool]]:
    """Assign each target a DISTINCT stable state, preferring assignments within tolerance.

    `errors[i, j]` = max abs error (over observed pops) between target i and stable state j.
    We use the Hungarian algorithm (linear_sum_assignment) on a modified cost: pairs outside
    `tol` get a large added penalty. A plain min-sum assignment can pick a solution with one
    out-of-tolerance pair even though another assignment has all pairs within tol (e.g.
    0.02+0.02 vs 0.0+0.03: the second has the lower sum but fails). With the penalty, the
    solver first minimizes the number of failures, so `success` is True iff SOME valid
    one-to-one matching within tolerance exists.

    Penalty derivation: any assignment pairs at most K targets, so its total real cost is at
    most K * max(errors). If the penalty P exceeds that, removing one penalized pair always
    saves more (P) than any possible increase in real cost (< P), so the number of
    out-of-tolerance pairs is minimized first. P = 1 + K * max(errors) satisfies this for
    any scale of errors; no assumption on r_max or on targets lying inside [0, r_max].

    Only the assignment uses the penalized cost; the returned errors are the REAL errors.
    """
    n_targets, n_states = errors.shape
    assignment: list[int | None] = [None] * n_targets
    if n_states > 0:
        penalty = 1.0 + n_targets * float(errors.max())
        cost = errors + penalty * (errors > tol)
        rows, cols = linear_sum_assignment(cost)  # rectangular OK: min(K, S) pairs
        for i, j in zip(rows, cols, strict=True):
            assignment[int(i)] = int(j)
    errs = [None if j is None else float(errors[i, j]) for i, j in enumerate(assignment)]
    matched = [e is not None and e <= tol for e in errs]
    return assignment, errs, matched


def _find_points(
    model: RateNetwork, net: NetArrays, targets: np.ndarray, cfg: CheckerConfig
) -> tuple[list[np.ndarray], bool]:
    """Dispatch to the exact N=1 search or the N>=2 search; also say if it is exhaustive."""
    if model.n_pops == 1:
        return [np.array([r]) for r in find_fixed_points_1d(net, cfg)], True
    return find_fixed_points_nd(model, net, targets, cfg), False


def find_stable_states(
    model: RateNetwork, params: Mapping[str, float], cfg: CheckerConfig | None = None
) -> list[FixedPoint]:
    """Stable fixed points of the network, without any targets (used to PLANT tasks).

    Task generation has no targets yet (it is choosing them), so the target-seeded local
    search is skipped: for N>=2 only the Latin-hypercube multi-start runs, whose cost is set
    by `cfg.n_starts` (a cheap prefilter uses few starts). Same classification as `check`.
    """
    cfg = cfg or CheckerConfig()
    net = model.net(params)
    no_targets = np.empty((0, len(model.observed)))
    points, _ = _find_points(model, net, no_targets, cfg)
    fps = [describe_fixed_point(model, net, r, cfg.eps_stab) for r in points]
    return [fp for fp in fps if fp.stable]


def check(
    model: RateNetwork,
    params: Mapping[str, float],
    targets: np.ndarray,
    cfg: CheckerConfig | None = None,
) -> CheckResult:
    """Decide whether `params` realize the K target states, and report how.

    `targets` has shape (K, n_observed): one row per target state, columns in the model's
    `observed` order. Success means every target is matched to a distinct STABLE fixed point
    within cfg.tol on every observed population (and, if configured, no extra stable states).

    This is the authoritative judge used for final reporting and by the expert objective;
    it is deterministic and independent of the simulator.
    """
    cfg = cfg or CheckerConfig()
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != len(model.observed):
        raise ValueError(f"targets must have shape (K, {len(model.observed)}), got {targets.shape}")

    net = model.net(params)
    points, exhaustive = _find_points(model, net, targets, cfg)
    fps = [describe_fixed_point(model, net, r, cfg.eps_stab) for r in points]
    stable_idx = [i for i, fp in enumerate(fps) if fp.stable]

    # errors[i, j]: target i vs j-th STABLE state, worst error over observed populations
    obs = np.array([fps[i].observed for i in stable_idx]).reshape(
        len(stable_idx), len(model.observed)
    )
    errors = np.abs(targets[:, None, :] - obs[None, :, :]).max(axis=2)
    assignment, errs, matched = _match(errors, cfg.tol)

    # Translate stable-list positions back to indices into `fps`.
    target_assignment = [None if j is None else stable_idx[j] for j in assignment]
    assigned = {a for a in target_assignment if a is not None}
    extra = [i for i in stable_idx if i not in assigned]  # not assigned to ANY target
    matched_fp = {a for a, m in zip(target_assignment, matched, strict=True) if m}

    success = all(matched) and not (cfg.forbid_extra_stable_states and len(extra) > 0)

    # Margins over matched states only.
    margin_eig = min((-fps[i].max_re for i in matched_fp), default=None)
    unstable = [np.array(fp.r) for fp in fps if not fp.stable]
    margin_dist = None
    if matched_fp and unstable:
        margin_dist = float(
            min(np.linalg.norm(np.array(fps[i].r) - u) for i in matched_fp for u in unstable)
        )

    warnings: list[str] = []
    if len(stable_idx) >= 2 and not unstable:
        # In 1D two stable points are always separated by an unstable one; in N-D a
        # separatrix (saddle) generally exists too. Finding none means the search missed it,
        # so margin_dist is None for the wrong reason.
        warnings.append(
            "two or more stable fixed points but no unstable one found: the global search "
            "most likely missed a saddle, so margin_dist is unavailable"
        )

    return CheckResult(
        fixed_points=fps,
        stable_indices=stable_idx,
        n_stable=len(stable_idx),
        target_assignment=target_assignment,
        target_errors=errs,
        target_matched=matched,
        extra_stable_indices=extra,
        n_extra_stable=len(extra),
        success=bool(success),
        margin_eig=margin_eig,
        margin_dist=margin_dist,
        global_search_exhaustive=exhaustive,
        warnings=warnings,
        tol=cfg.tol,
    )
