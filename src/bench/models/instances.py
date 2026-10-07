"""Concrete models: a builder function per model + a loader from YAML config.

A builder maps the FULL parameter dict (free + fixed) to `NetArrays`. All parameters
are positive (Dale's law); the sign of each connection is applied here, in the rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bench.models.base import NetArrays, ParamSpace, ParamSpec
from bench.models.rate_network import RateNetwork


def build_toy_1pop(p: Mapping[str, float]) -> NetArrays:
    """Structure rule for the 1-population toy: W = [[w]], everything else copied through.

    theta is a FIXED parameter: in 1D only I - theta matters, so leaving both free would
    give the optimizer a redundant direction.

    The toy is the simplest bistable circuit: one excitatory population exciting
    itself with strength w. If w is strong enough relative to the sigmoid width k,
    the feedback can hold the population either "off" (low rate) or "on" (high rate).
    Because it is one-dimensional, the checker can find ALL its fixed points exactly,
    which makes it the ground truth for validating the machinery used on bigger models.

    `p` is the full parameter dict (free + fixed). w is positive because the
    connection is excitatory; Dale's law needs no sign flip here.
    """
    return NetArrays(
        W=np.array([[p["w"]]]),
        I_ext=np.array([p["I"]]),
        tau=np.array([p["tau"]]),
        theta=np.array([p["theta"]]),
        k=np.array([p["k"]]),
        r_max=np.array([p["r_max"]]),
    )


def build_ei_2pop(p: Mapping[str, float]) -> NetArrays:
    """Structure rule for the Wilson-Cowan-style E/I pair (population order: E, I).

    Four connection strengths, all stored as positive numbers (Dale's law), with the sign
    applied here. W[i, j] is the weight from j onto i:

                   from E     from I
        onto E     +w_EE      -w_EI
        onto I     +w_IE      -w_II

    Excitation (columns of E) is positive, inhibition (column of I) is negative. W is not
    symmetric and the two populations have different time constants, which is what lets
    E/I circuits oscillate as well as be multistable; the checker must cope with both.
    `I_E`, `I_I` are the external drives. Thresholds, widths, taus and r_max are fixed
    (one theta per population is redundant with the drive, as in the toy).
    """
    return NetArrays(
        W=np.array([[p["w_EE"], -p["w_EI"]], [p["w_IE"], -p["w_II"]]]),
        I_ext=np.array([p["I_E"], p["I_I"]]),
        tau=np.array([p["tau_E"], p["tau_I"]]),
        theta=np.array([p["theta_E"], p["theta_I"]]),
        k=np.array([p["k_E"], p["k_I"]]),
        r_max=np.array([p["r_max_E"], p["r_max_I"]]),
    )


def build_wc1972(p: Mapping[str, float]) -> NetArrays:
    """Structure rule for Wilson & Cowan (1972), eqs. 11, 12 and 15 (population order: E, I).

        tau_e dE/dt = -E + (k_e - r_e E) S_e(c1 E - c2 I + P)
        tau_i dI/dt = -I + (k_i - r_i I) S_i(c3 E - c4 I + Q)
        S(x) = 1 / (1 + exp(-a (x - theta))) - 1 / (1 + exp(a theta))

    Parameters (names as in the model card): c1..c4 (synapse counts, all positive; the signs
    are applied here), a_e, theta_e, a_i, theta_i (slope and position of the sigmoids), inputs
    P and Q, time constants tau_e, tau_i, refractory coefficients rref_e, rref_i (the paper's
    r_e, r_i, set to 1). k_e, k_i, the maximum of the response, are not given by the paper as a
    formula; by default k = S(+inf) = 1 - 1/(1 + exp(a theta)) (the card's interpretation). Put
    `k_e` and `k_i` among the fixed parameters to use another value (the card's alternative is 1).

    Mapping onto NetArrays: the sigmoid width is k = 1 / a, the amplitude `r_max` is the card's
    k, and `rref` is r. The shifted sigmoid is the "wc_shifted" variant.
    """
    a = np.array([p["a_e"], p["a_i"]])
    theta = np.array([p["theta_e"], p["theta_i"]])
    k_card = np.array([p["k_e"], p["k_i"]]) if "k_e" in p else 1.0 - 1.0 / (1.0 + np.exp(a * theta))
    return NetArrays(
        W=np.array([[p["c1"], -p["c2"]], [p["c3"], -p["c4"]]]),
        I_ext=np.array([p["P"], p["Q"]]),
        tau=np.array([p["tau_e"], p["tau_i"]]),
        theta=theta,
        k=1.0 / a,
        r_max=k_card,
        rref=np.array([p["rref_e"], p["rref_i"]]),
        variant="wc_shifted",
    )


# Placeholder (NOT implemented, out of scope for now): 3-pop selective model with
# populations (E1, E2, I) and structured parameters w_plus (same-selectivity E->E),
# w_minus (cross-selectivity E->E), plus the E->I / I->E / I->I weights. It will be a
# new `build_selective_3pop` registered below, with its own YAML in configs/.

BUILDERS: dict[str, Callable[[Mapping[str, float]], NetArrays]] = {
    "toy_1pop": build_toy_1pop,
    "ei_2pop": build_ei_2pop,
    "wc1972": build_wc1972,
}


def model_from_dict(cfg: Mapping[str, Any]) -> RateNetwork:
    """Turn a parsed YAML model description into a `RateNetwork`.

    This is how model configs (populations, observed pops, parameter bounds, fixed
    values, which builder rule to use) become a usable model object. It validates
    eagerly: unknown keys, unknown builders and bad bounds raise immediately, so a
    typo in a config fails at load time instead of producing a silently wrong model.
    """
    allowed = {"name", "builder", "populations", "types", "observed", "free", "fixed"}
    extra = set(cfg) - allowed
    if extra:
        raise ValueError(f"unknown model config keys: {sorted(extra)}")
    if cfg["builder"] not in BUILDERS:
        raise ValueError(f"unknown builder {cfg['builder']!r}; known: {sorted(BUILDERS)}")
    free = tuple(
        ParamSpec(name, float(s["low"]), float(s["high"]), bool(s.get("log", False)))
        for name, s in cfg["free"].items()
    )
    fixed = {k: float(v) for k, v in cfg["fixed"].items()}
    return RateNetwork(
        name=cfg["name"],
        populations=tuple(cfg["populations"]),
        types=tuple(cfg["types"]),
        observed=tuple(cfg["observed"]),
        space=ParamSpace(free, fixed),
        build=BUILDERS[cfg["builder"]],
    )


def load_model(path: str | Path) -> RateNetwork:
    """Read a model YAML file from disk and build it (see `model_from_dict`)."""
    with open(path) as f:
        return model_from_dict(yaml.safe_load(f))
