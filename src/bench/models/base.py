"""Parameter spaces and the plain-array description of a network.

Nothing here knows about dynamics; it is only bookkeeping for parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

Params = dict[str, float]


@dataclass(frozen=True)
class NetArrays:
    """Everything the dynamics need, for N populations.

    W[i, j] is the weight from population j onto population i, so u = W @ r + I_ext.
    Dale's law is already applied: columns of inhibitory populations are <= 0.

    The dynamics are  tau_i dr_i/dt = -r_i + (amp_i - rref_i * r_i) * S_i(u_i)  where
      * S is a sigmoid of the input, chosen by `variant`:
          "logistic"   S(u) = 1 / (1 + exp(-(u - theta) / k))                     (0 to 1)
          "wc_shifted" S(u) = 1 / (1 + exp(-(u - theta) / k)) - 1 / (1 + exp(theta / k))
                       (Wilson & Cowan 1972, eq. 15 with a = 1/k: shifted so S(0) = 0);
      * amp is the field `r_max` (kept under its old name for backward compatibility; for the
        logistic model it IS the maximum rate; for Wilson-Cowan it is the card's k_e, k_i);
      * rref >= 0 is the refractory coefficient (the card's r_e, r_i). With rref = 0 the
        model is the plain rate network  tau dr/dt = -r + r_max * S(u)  used so far.
    The sigmoid width `k` here is NOT the Wilson-Cowan k: it is 1 / a.
    """

    W: np.ndarray  # (N, N)
    I_ext: np.ndarray  # (N,)
    tau: np.ndarray  # (N,)
    theta: np.ndarray  # (N,)
    k: np.ndarray  # (N,) sigmoid width (1 / slope a)
    r_max: np.ndarray  # (N,) amplitude of the response: max rate (logistic) / card's k_i (WC)
    rref: np.ndarray | None = None  # (N,) refractory coefficient; None = zeros
    variant: str = "logistic"  # "logistic" | "wc_shifted"
    shift: np.ndarray = field(init=False, repr=False, compare=False)  # derived: S = sigma - shift

    def __post_init__(self) -> None:
        if self.variant not in ("logistic", "wc_shifted"):
            raise ValueError(f"unknown sigmoid variant {self.variant!r}")
        zeros = np.zeros_like(np.asarray(self.r_max, dtype=float))
        object.__setattr__(self, "rref", zeros if self.rref is None else np.asarray(self.rref))
        if self.variant == "wc_shifted":  # sigma(-theta / k): the logistic's value at u = 0
            shift = 1.0 / (1.0 + np.exp(self.theta / self.k))
        else:
            shift = zeros
        object.__setattr__(self, "shift", shift)


# Absolute slack allowed at the bounds (round-off from float arithmetic, e.g. a value
# produced by from_unit(1.0) or by a config round trip) before to_unit raises.
BOUND_TOL = 1e-12


@dataclass(frozen=True)
class ParamSpec:
    """One free parameter: bounds and whether it is searched on a log scale."""

    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.name}: need low < high, got [{self.low}, {self.high}]")
        if self.log and self.low <= 0:
            raise ValueError(f"{self.name}: log-scale parameter needs low > 0")

    def to_unit(self, x: float) -> float:
        """Real value -> position in [0, 1] (linear, or log-spaced if `log`).

        Raises if x is outside the bounds by more than BOUND_TOL, so a genuinely bad
        value is caught here; round-off-sized excursions are tolerated and clipped.
        """
        if not (self.low - BOUND_TOL <= x <= self.high + BOUND_TOL):
            raise ValueError(f"{self.name}={x} outside bounds [{self.low}, {self.high}]")
        if self.log:
            u = float(np.log(x / self.low) / np.log(self.high / self.low))
        else:
            u = (x - self.low) / (self.high - self.low)
        return min(max(u, 0.0), 1.0)

    def from_unit(self, u: float) -> float:
        """Position in [0, 1] -> real value; the inverse of `to_unit`.

        On a log scale equal steps in u are equal *ratios* in the value, which suits
        parameters like a sigmoid width k whose useful range spans an order of magnitude.
        """
        # Clip: optimizers may overshoot [0, 1] slightly; the box is a hard constraint.
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            return float(self.low * (self.high / self.low) ** u)
        return float(self.low + u * (self.high - self.low))


@dataclass(frozen=True)
class ParamSpace:
    """Free parameters (searched) plus fixed parameters (constants of the model)."""

    free: tuple[ParamSpec, ...]
    fixed: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [p.name for p in self.free]
        if len(set(names)) != len(names):
            raise ValueError("duplicate free parameter names")
        both = set(names) & set(self.fixed)
        if both:
            raise ValueError(f"parameters both free and fixed: {sorted(both)}")
        # Read-only view: the Environment hands out its ParamSpace, and a mutable `fixed` dict
        # would let a caller change the model's constants (tau, theta, ...) from outside.
        object.__setattr__(self, "fixed", MappingProxyType(dict(self.fixed)))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.free)

    @property
    def dim(self) -> int:
        return len(self.free)

    def to_unit(self, params: Mapping[str, float]) -> np.ndarray:
        """Free params (dict) -> point in the unit cube [0, 1]^d, ordered as `names`.

        Optimizers will work in this normalized space so that every parameter has the
        same scale regardless of its physical range. `params` must contain exactly the
        free parameters (fixed ones are not allowed here).
        """
        self._check_names(params)
        return np.array([p.to_unit(params[p.name]) for p in self.free])

    def from_unit(self, u: np.ndarray) -> Params:
        """Point in the unit cube -> dict of free params; the inverse of `to_unit`.

        This is how an optimizer's proposal becomes model parameters. Coordinates
        slightly outside [0, 1] are clipped, since the bounds are a hard constraint.
        """
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {u.shape}")
        return {p.name: p.from_unit(ui) for p, ui in zip(self.free, u, strict=True)}

    def full(self, params: Mapping[str, float]) -> Params:
        """Merge the searched (free) parameters with the model's constants (fixed).

        `build` needs every parameter, but optimizers and tasks only deal with the free
        ones; this keeps that separation and rejects missing or unknown names.
        """
        self._check_names(params)
        return {**self.fixed, **{n: float(params[n]) for n in self.names}}

    def _check_names(self, params: Mapping[str, float]) -> None:
        if set(params) != set(self.names):
            raise ValueError(f"expected free params {sorted(self.names)}, got {sorted(params)}")
