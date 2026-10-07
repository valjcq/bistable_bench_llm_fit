"""Fixed-step RK4 simulation of the rate network.

The simulator is deliberately independent of the checker: it only integrates the ODE.
Its role is (a) the cheap "naive" objective (simulate and look where the system ends up)
and (b) an independent cross-check of the checker's fixed points in the tests.

Fixed step + plain floating-point arithmetic => bit-reproducible results. Many initial
conditions can be integrated in one vectorized call (`simulate_batch`); each trajectory is
identical, bit for bit, to what a single-IC call would give.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from bench.models.base import NetArrays
from bench.models.rate_network import RateNetwork, rhs


@dataclass(frozen=True)
class SimConfig:
    """Integration settings."""

    dt: float = 0.1  # RK4 step. Must be well below the smallest tau (tau ~ 5-10 here).
    flow_tol: float = 1e-6  # "at rest" means max_i |dr_i/dt| < flow_tol
    # The system must stay at rest for hold_tau_factor * max(tau). The slowest relaxation
    # time scale of the network is of order max(tau), so a window of a few max(tau)
    # separates a real fixed point from a momentary slow-down near a saddle.
    hold_tau_factor: float = 5.0

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.flow_tol <= 0 or self.hold_tau_factor <= 0:
            raise ValueError("invalid SimConfig")


@dataclass(frozen=True)
class SimResult:
    """Summary of one trajectory (not the trajectory itself). JSON-serializable."""

    final_state: list[float]  # state when the run stopped (settled, or end time T)
    settled: bool  # True if it came to rest (flow < flow_tol for the hold window) before T
    time_to_settle: float | None  # time the sustained rest began; None if never settled
    t_end: float  # time at which the run stopped
    n_steps: int


def _rk4_step(r: np.ndarray, net: NetArrays, dt: float) -> np.ndarray:
    """One classical 4th-order Runge-Kutta step of dr/dt = rhs(r); r is (N,) or (n, N)."""
    k1 = rhs(r, net)
    k2 = rhs(r + 0.5 * dt * k1, net)
    k3 = rhs(r + 0.5 * dt * k2, net)
    k4 = rhs(r + dt * k3, net)
    return r + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_batch(
    model: RateNetwork,
    params: Mapping[str, float],
    x0s: np.ndarray,
    T: float,
    cfg: SimConfig | None = None,
) -> list[SimResult]:
    """Integrate the network from each row of `x0s` (shape (n_ic, N)) for at most time T.

    One vectorized RK4 loop advances all initial conditions together, which is what makes
    multi-IC checks (naive objective, cross-checks) affordable in pure numpy. A trajectory is
    frozen as soon as it has been at rest for the hold window (hold_tau_factor * max tau), so
    its result does not depend on the other rows or on how long they keep running.
    A trajectory still moving at T gets settled=False: callers must treat that as "did not
    reach a fixed point" (slow transient near a saddle, or a limit cycle).
    """
    cfg = cfg or SimConfig()
    net = model.net(params)
    R = np.array(x0s, dtype=float)  # copy: we update rows in place
    if R.ndim != 2 or R.shape[1] != model.n_pops:
        raise ValueError(f"x0s must have shape (n_ic, {model.n_pops}), got {R.shape}")
    if T <= 0:
        raise ValueError("T must be positive")

    n_ic = R.shape[0]
    n_steps = int(np.ceil(T / cfg.dt - 1e-12))
    hold_steps = int(np.ceil(cfg.hold_tau_factor * float(net.tau.max()) / cfg.dt))
    rest_since = np.full(n_ic, -1)  # step at which the current rest period began, -1 = moving
    stop_step = np.full(n_ic, n_steps)
    settled = np.zeros(n_ic, dtype=bool)
    active = np.ones(n_ic, dtype=bool)  # still being integrated

    for step in range(n_steps):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        at_rest = np.abs(rhs(R[idx], net)).max(axis=1) < cfg.flow_tol
        rest_since[idx[at_rest & (rest_since[idx] < 0)]] = step  # rest period starts now
        rest_since[idx[~at_rest]] = -1  # moved again: restart the window
        done = at_rest & (step - rest_since[idx] >= hold_steps)
        settled[idx[done]] = True
        stop_step[idx[done]] = step
        active[idx[done]] = False
        keep = idx[~done]
        R[keep] = _rk4_step(R[keep], net, cfg.dt)

    return [
        SimResult(
            final_state=[float(x) for x in R[i]],
            settled=bool(settled[i]),
            time_to_settle=float(rest_since[i] * cfg.dt) if settled[i] else None,
            t_end=float(stop_step[i] * cfg.dt),
            n_steps=int(stop_step[i]),
        )
        for i in range(n_ic)
    ]


def simulate(
    model: RateNetwork,
    params: Mapping[str, float],
    x0: np.ndarray,
    T: float,
    cfg: SimConfig | None = None,
) -> SimResult:
    """Simulate one initial condition `x0` (shape (N,)); see `simulate_batch`."""
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (model.n_pops,):
        raise ValueError(f"x0 must have shape ({model.n_pops},), got {x0.shape}")
    return simulate_batch(model, params, x0[None, :], T, cfg)[0]
