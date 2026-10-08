"""Generic N-population rate network.

    tau_i dr_i/dt = -r_i + F_i(u_i),    u = W r + I_ext
    F_i(u) = r_max_i / (1 + exp(-(u - theta_i) / k_i))

The functions below are pure: they take a `NetArrays` and never touch parameters
directly. `RateNetwork` adds the model-specific part (names, bounds, `build`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import expit

from bench.models.base import NetArrays, ParamSpace, TaskInputSpec

BuildFn = Callable[[Mapping[str, float]], NetArrays]


def sigmoid(u: np.ndarray, net: NetArrays) -> np.ndarray:
    """The sigmoid S(u) of each population (variant "logistic" or "wc_shifted", see NetArrays).

    S = expit((u - theta) / k) - shift, where shift is 0 for the logistic and
    1 / (1 + exp(theta / k)) for the Wilson-Cowan shifted form (eq. 15), whose value at u = 0
    is exactly 0. `expit` is scipy's overflow-safe logistic, so huge |u| gives the asymptote.
    """
    return expit((u - net.theta) / net.k) - net.shift


def _amplitude(net: NetArrays, r: np.ndarray | None) -> np.ndarray:
    """amp - rref * r: the room left for firing (the paper's refractory factor, k - r E).

    `r` is only needed when some rref is nonzero; the plain rate network (rref = 0) can be
    evaluated from the input alone, which keeps the old call signatures valid.
    """
    if r is None:
        if np.any(net.rref):
            raise ValueError("the state r is required when the refractory coefficient is nonzero")
        return net.r_max
    return net.r_max - net.rref * r


def rate_fn(u: np.ndarray, net: NetArrays, r: np.ndarray | None = None) -> np.ndarray:
    """Response of each population to its total input u: F = (amp - rref * r) * S(u).

    S is a sigmoid between its asymptotes; theta is the input at which it switches (maximum
    slope) and k the width. This is the only nonlinearity in the model, and it is what makes
    multistability possible: a sigmoid fed back through a strong enough positive loop can cross
    the identity line several times. With rref = 0 (the logistic rate network) F = r_max * S(u)
    depends on u alone and `r` may be omitted; with a refractory term (Wilson-Cowan) the state
    `r` must be given because F then also depends on the population's own rate.

    `u` has shape (N,) (or (..., N)), one input per population. Returns the same shape.
    """
    return _amplitude(net, r) * sigmoid(u, net)


def rate_deriv(u: np.ndarray, net: NetArrays, r: np.ndarray | None = None) -> np.ndarray:
    """Slope of the response with respect to its input, dF/du = (amp - rref * r) * S'(u).

    Used to build the Jacobian: it says how strongly a small change in input u changes the
    output, i.e. the "gain" of each population at the current operating point. For the logistic
    it peaks at r_max / (4k) when u = theta and vanishes when the population is saturated;
    saturated populations are what make the stable states of a bistable network stable.

    Analytic form: with s = logistic((u - theta) / k), S' = s * (1 - s) / k (the shift of the
    Wilson-Cowan sigmoid is a constant, so it does not change the slope). Tested against finite
    differences in tests/test_model.py.
    """
    s = expit((u - net.theta) / net.k)
    return _amplitude(net, r) / net.k * s * (1.0 - s)


def rhs(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """Right-hand side of the dynamics: dr_i/dt = (-r_i + F_i) / tau_i.

    This is what the simulator integrates. Each population relaxes toward the rate
    its input demands, F = (amp - rref r) S(W r + I_ext), at a speed set by its time constant
    tau. Where this is zero for every population, r is a fixed point.

    `r` may be one state, shape (N,), or a batch of states, shape (n, N): the simulator
    integrates many initial conditions at once. The input u_i = sum_j W_ij r_j is written as
    an explicit multiply-and-sum rather than a matrix product so that every row of a batch
    goes through exactly the same arithmetic as a lone state, which keeps a trajectory's bits
    independent of what else is in the batch (BLAS may pick different kernels per shape).
    """
    u = (net.W * r[..., None, :]).sum(axis=-1) + net.I_ext
    return (-r + rate_fn(u, net, r)) / net.tau


def fixed_point_residual(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """g(r) = -r + F(W r + I_ext, r): the dynamics without the 1/tau factor.

    The checker looks for roots of g, i.e. the fixed points. tau only rescales how
    fast the system moves, never where it rests, so dropping it keeps the root
    finding independent of the time constants (and better scaled numerically).
    For N=1 the checker scans this function along the state range for sign changes.
    """
    return -r + rate_fn(net.W @ r + net.I_ext, net, r)


def residual_jacobian(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """Derivative of the residual g: dg/dr = -Id + diag(dF/du) @ W - diag(rref * S(u)).

    Handed to Newton's method (scipy.optimize.root) so it can converge quickly to a
    fixed point without estimating derivatives numerically. Row i of the product is
    dF/du_i times row i of W: population i's gain scales *its own* inputs, which is
    why the gain multiplies W from the left (row scaling), not the right. The last term only
    exists with a refractory coefficient: F_i also depends on r_i directly, through the factor
    (amp_i - rref_i r_i), with derivative -rref_i S(u_i). For Wilson-Cowan this is
    J11 = [-1 - r_e S_e + (k_e - r_e E) c1 S_e'] / tau_e (model card, "Jacobian").
    """
    u = net.W @ r + net.I_ext
    jac = -np.eye(len(r)) + rate_deriv(u, net, r)[:, None] * net.W
    if np.any(net.rref):
        jac = jac - np.diag(net.rref * sigmoid(u, net))
    return jac


def jacobian(r: np.ndarray, net: NetArrays) -> np.ndarray:
    """Jacobian of the dynamics at state r: J = diag(1/tau) @ dg/dr.

    This decides whether a fixed point is stable. Near a fixed point the system
    behaves like dx/dt = J x, so if every eigenvalue of J has negative real part,
    small perturbations die out (stable); if any has positive real part they grow
    (unstable). The checker computes eigenvalues of this matrix, and the margin
    score is built from how negative they are. It is only meaningful at a fixed
    point; elsewhere it is just the local slope of the flow.
    """
    return residual_jacobian(r, net) / net.tau[:, None]


def state_bounds(net: NetArrays) -> tuple[np.ndarray, np.ndarray]:
    """Per-population bounds (lo, hi) of every state the dynamics can occupy.

    Every fixed point lies inside [lo, hi], and a trajectory that starts inside stays inside.
    Derivation. The response is F = (amp - rref r) S(u) with the sigmoid S confined to the open
    interval (S_min, S_max) = (-shift, 1 - shift) by its asymptotes (shift = 0 for the
    logistic). At a steady state r = (amp - rref r) S, i.e. r = amp S / (1 + rref S), which is
    increasing in S (derivative amp / (1 + rref S)^2 > 0 for rref S > -1). Hence
        lo = amp S_min / (1 + rref S_min),    hi = amp S_max / (1 + rref S_max).
    Invariance: at r = hi the flow is -hi + (amp - rref hi) S(u) <= -hi + (amp - rref hi) S_max
    = 0, and symmetrically >= 0 at r = lo, so the box cannot be left (amp - rref r stays > 0
    inside it for the shipped parameters, amp > rref hi).
    Cases.
      * Logistic (rref = 0, shift = 0): [0, amp], the box [0, r_max] used so far.
      * Wilson-Cowan with k = S(+inf) = S_max (the card's interpretation): hi = k^2 / (1 + k),
        about 0.48 for the card's sets, and lo slightly negative (E, I may dip below their
        background; the card calls small negative values "depression below background").
      * Wilson-Cowan with k = 1: S(+inf) < 1, hi = S_max / (1 + S_max) (a bit below 1/2). We use
        the true asymptote S(+inf) rather than k, which keeps the bound tight; k itself would
        also be valid but slightly loose.
    """
    s_lo = 0.0 - net.shift  # (0.0 - 0.0 keeps +0.0 for the logistic, never -0.0)
    s_hi = 1.0 - net.shift
    lo = net.r_max * s_lo / (1.0 + net.rref * s_lo)
    hi = net.r_max * s_hi / (1.0 + net.rref * s_hi)
    return lo, hi


@dataclass(frozen=True)
class RateNetwork:
    name: str
    populations: tuple[str, ...]
    types: tuple[str, ...]  # "E" or "I", same order as populations
    observed: tuple[str, ...]  # subset of populations that targets refer to
    space: ParamSpace
    build: BuildFn  # full params (free + fixed) -> NetArrays, applies structure + Dale signs
    # Range of observed rates that "typical" targets live in. Set by the config (not derived from
    # the parameters): the tolerance is a fraction of its width and the planting window a fraction
    # of it. The logistic models use [0, 1] (= [0, r_max]).
    reference_range: tuple[float, float] = (0.0, 1.0)
    # External inputs that belong to the task (see TaskInputSpec); empty for the logistic models.
    task_inputs: tuple[TaskInputSpec, ...] = ()

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

    def with_inputs(self, inputs: Mapping[str, float]) -> RateNetwork:
        """The same model with the task's external inputs set (they override the fixed values).

        `inputs` must hold exactly the declared task inputs, each inside its allowed range.
        """
        specs = {s.name: s for s in self.task_inputs}
        if set(inputs) != set(specs):
            raise ValueError(f"expected task inputs {sorted(specs)}, got {sorted(inputs)}")
        for name, value in inputs.items():
            if not specs[name].contains(value):
                raise ValueError(f"task input {name}={value} outside its allowed range")
        fixed = {**self.space.fixed, **{k: float(v) for k, v in inputs.items()}}
        return replace(self, space=ParamSpace(self.space.free, fixed))

    def state_bounds(self, params: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Model-provided state bounds (lo, hi) per population for these parameters.

        Every place that used to assume the state lies in [0, r_max] (the 1-D scan, the global
        search box, the unobserved-population solve, initial conditions of the naive objective,
        the planting window) asks this instead. See `state_bounds` for the derivation.
        """
        return state_bounds(self.net(params))

    def net(self, params: Mapping[str, float]) -> NetArrays:
        """Turn a dict of the FREE parameters into the arrays the dynamics need.

        This is the single entry point from "parameters an optimizer proposes" to
        "numbers the model uses": it adds the fixed parameters, then calls `build`,
        which applies the connectivity structure and Dale's law signs. Everything
        downstream (checker, simulator) starts from the NetArrays returned here.
        """
        return self.build(self.space.full(params))
