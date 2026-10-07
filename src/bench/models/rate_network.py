"""Generic N-population rate network.

    tau_i dr_i/dt = -r_i + F_i(u_i),    u = W r + I_ext
    F_i(u) = r_max_i / (1 + exp(-(u - theta_i) / k_i))

The functions below are pure: they take a `NetArrays` and never touch parameters
directly. `RateNetwork` adds the model-specific part (names, bounds, `build`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from bench.models.base import NetArrays, ParamSpace

BuildFn = Callable[[Mapping[str, float]], NetArrays]


def rate_fn(u: np.ndarray, net: NetArrays) -> np.ndarray:
    """Firing-rate nonlinearity: turn each population's total input u into a rate.

    F(u) = r_max / (1 + exp(-(u - theta) / k)), a sigmoid between 0 and r_max.
    theta is the input at which the rate is half of r_max; k sets how sharp the
    switch is (small k = nearly a step). This is the only nonlinearity in the
    model, and it is what makes multistability possible: a sigmoid fed back
    through a strong enough positive loop can cross the identity line several times.

    `u` has shape (N,), one input per population. Returns shape (N,).
    `expit` is scipy's overflow-safe logistic, so huge |u| gives 0 or r_max, not nan.
    """
    return net.r_max * expit((u - net.theta) / net.k)


def rate_deriv(u: np.ndarray, net: NetArrays) -> np.ndarray:
    """Slope of the sigmoid, F'(u), evaluated per population.

    Used to build the Jacobian: F' says how strongly a small change in input u
    changes the output rate, i.e. the "gain" of each population at the current
    operating point. It peaks at r_max / (4k) when u = theta and vanishes when the
    population is saturated (near 0 or r_max); saturated populations are what make
    the stable states of a bistable network stable.

    Analytic form: with s = logistic((u - theta) / k), F' = (r_max / k) * s * (1 - s).
    Tested against finite differences in tests/test_model.py.
    """
    s = expit((u - net.theta) / net.k)
    return net.r_max / net.k * s * (1.0 - s)


def rhs(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """Right-hand side of the dynamics: dr_i/dt = (-r_i + F_i(u_i)) / tau_i.

    This is what the simulator integrates. Each population relaxes toward the rate
    its input demands, F(W r + I_ext), at a speed set by its time constant tau.
    Where this is zero for every population, r is a fixed point.

    `r` may be one state, shape (N,), or a batch of states, shape (n, N): the simulator
    integrates many initial conditions at once. The input u_i = sum_j W_ij r_j is written as
    an explicit multiply-and-sum rather than a matrix product so that every row of a batch
    goes through exactly the same arithmetic as a lone state, which keeps a trajectory's bits
    independent of what else is in the batch (BLAS may pick different kernels per shape).
    """
    u = (net.W * r[..., None, :]).sum(axis=-1) + net.I_ext
    return (-r + rate_fn(u, net)) / net.tau


def fixed_point_residual(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """g(r) = -r + F(W r + I_ext): the dynamics without the 1/tau factor.

    The checker looks for roots of g, i.e. the fixed points. tau only rescales how
    fast the system moves, never where it rests, so dropping it keeps the root
    finding independent of the time constants (and better scaled numerically).
    For N=1 the checker scans this function along [0, r_max] for sign changes.
    """
    return -r + rate_fn(net.W @ r + net.I_ext, net)


def residual_jacobian(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """Derivative of the residual g: dg/dr = -Id + diag(F'(u)) @ W.

    Handed to Newton's method (scipy.optimize.root) so it can converge quickly to a
    fixed point without estimating derivatives numerically. Row i of the product is
    F'(u_i) times row i of W: population i's gain scales *its own* inputs, which is
    why the gain multiplies W from the left (row scaling), not the right.
    """
    u = net.W @ r + net.I_ext
    return -np.eye(len(r)) + rate_deriv(u, net)[:, None] * net.W


def jacobian(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """Jacobian of the dynamics at state r: J = diag(1/tau) @ (-Id + diag(F'(u)) @ W).

    This decides whether a fixed point is stable. Near a fixed point the system
    behaves like dx/dt = J x, so if every eigenvalue of J has negative real part,
    small perturbations die out (stable); if any has positive real part they grow
    (unstable). The checker computes eigenvalues of this matrix, and the margin
    score is built from how negative they are. It is only meaningful at a fixed
    point; elsewhere it is just the local slope of the flow.
    """
    return residual_jacobian(r, net) / net.tau[:, None]


@dataclass(frozen=True)
class RateNetwork:
    name: str
    populations: tuple[str, ...]
    types: tuple[str, ...]  # "E" or "I", same order as populations
    observed: tuple[str, ...]  # subset of populations that targets refer to
    space: ParamSpace
    build: BuildFn  # full params (free + fixed) -> NetArrays, applies structure + Dale signs

    def __post_init__(self) -> None:
        if len(self.types) != len(self.populations):
            raise ValueError("types and populations must have the same length")
        if not set(self.types) <= {"E", "I"}:
            raise ValueError(f"types must be 'E' or 'I', got {self.types}")
        if not self.observed or not set(self.observed) <= set(self.populations):
            raise ValueError("observed must be a non-empty subset of populations")

    @property
    def n_pops(self) -> int:
        """Number of populations N (1 for the toy, 2 for E/I)."""
        return len(self.populations)

    @property
    def observed_idx(self) -> np.ndarray:
        """Positions of the observed populations inside the full state vector.

        Targets only specify rates for observed populations (partial observation),
        so every comparison between a state and a target uses `r[observed_idx]`.
        """
        return np.array([self.populations.index(p) for p in self.observed])

    def net(self, params: Mapping[str, float]) -> NetArrays:
        """Turn a dict of the FREE parameters into the arrays the dynamics need.

        This is the single entry point from "parameters an optimizer proposes" to
        "numbers the model uses": it adds the fixed parameters, then calls `build`,
        which applies the connectivity structure and Dale's law signs. Everything
        downstream (checker, simulator) starts from the NetArrays returned here.
        """
        return self.build(self.space.full(params))
