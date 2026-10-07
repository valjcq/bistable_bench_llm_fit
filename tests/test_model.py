"""Tests 1-2: hand-checked toy case, and analytic derivatives vs finite differences."""

import numpy as np
import pytest
from scipy.optimize import brentq

from bench.models.base import ParamSpec
from bench.models.rate_network import jacobian, rate_deriv, rate_fn, residual_jacobian, rhs

TOY_BISTABLE = {"w": 10.0, "I": 0.0, "k": 1.0}  # theta = 5 is fixed in the config


def _fixed_points_1d(toy, params):
    """Find the toy's fixed points by a plain grid scan + brentq.

    Deliberately simple and independent of the future checker, so the hand-checked
    test below does not depend on the code it will later help validate.
    """
    net = toy.net(params)
    g = lambda r: -r + rate_fn(net.W[0, 0] * np.array(r) + net.I_ext, net)[0]  # noqa: E731
    grid = np.linspace(0.0, 1.0, 2000)  # avoids hitting r=0.5 exactly (g=0 there)
    vals = [g(r) for r in grid]
    return [
        brentq(g, a, b)
        for a, b, ga, gb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:], strict=True)
        if ga * gb < 0
    ]


def test_toy_bistable_hand_checked(toy):
    roots = _fixed_points_1d(toy, TOY_BISTABLE)
    assert len(roots) == 3
    # w=10, theta=5: r=0.5 gives u=5=theta, so F=0.5 -> fixed point by symmetry.
    assert roots[1] == pytest.approx(0.5, abs=1e-9)
    assert roots[0] == pytest.approx(0.007, abs=1e-3)
    assert roots[2] == pytest.approx(0.993, abs=1e-3)
    net = toy.net(TOY_BISTABLE)
    # Slope of g at each root: stability <=> eigenvalue of J < 0.
    eig = [jacobian(np.array([r]), net)[0, 0] for r in roots]
    assert eig[0] < 0 and eig[1] > 0 and eig[2] < 0


def test_toy_small_w_is_monostable(toy):
    # max slope of F is r_max/(4k) = 0.25, so w=2 -> w*0.25 = 0.5 < 1: g is monotone.
    params = {"w": 2.0, "I": 1.0, "k": 1.0}
    assert len(_fixed_points_1d(toy, params)) == 1


def _random_params(model, rng):
    """Draw a random point of the model's free-parameter box (via unit space)."""
    return model.space.from_unit(rng.random(model.space.dim))


def test_rate_deriv_matches_finite_differences(toy):
    rng = np.random.default_rng(0)
    for _ in range(10):
        net = toy.net(_random_params(toy, rng))
        u = rng.uniform(-5, 15, size=net.W.shape[0])
        h = 1e-6
        fd = (rate_fn(u + h, net) - rate_fn(u - h, net)) / (2 * h)
        np.testing.assert_allclose(rate_deriv(u, net), fd, rtol=1e-6, atol=1e-9)


def test_jacobian_matches_finite_differences(toy):
    rng = np.random.default_rng(1)
    for _ in range(10):
        net = toy.net(_random_params(toy, rng))
        r = rng.uniform(0, 1, size=net.W.shape[0])
        h = 1e-6
        fd = np.empty((len(r), len(r)))
        for j in range(len(r)):
            e = np.zeros(len(r))
            e[j] = h
            fd[:, j] = (rhs(r + e, net) - rhs(r - e, net)) / (2 * h)
        np.testing.assert_allclose(jacobian(r, net), fd, rtol=1e-5, atol=1e-9)


def test_unit_roundtrip(toy):
    rng = np.random.default_rng(2)
    u = rng.random(toy.space.dim)
    np.testing.assert_allclose(toy.space.to_unit(toy.space.from_unit(u)), u, atol=1e-12)


def test_jacobian_is_residual_jacobian_scaled_by_inverse_tau(toy):
    """The two Jacobians must differ only by the row scaling 1/tau (see their docstrings)."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        net = toy.net(_random_params(toy, rng))
        r = rng.uniform(0, 1, size=net.W.shape[0])
        np.testing.assert_allclose(
            jacobian(r, net), np.diag(1.0 / net.tau) @ residual_jacobian(r, net), rtol=1e-12
        )


def test_to_unit_bound_tolerance():
    spec = ParamSpec("x", 1.0, 5.0)
    assert spec.to_unit(5.0 + 1e-13) == 1.0  # round-off excursion: clipped, not an error
    assert spec.to_unit(1.0 - 1e-13) == 0.0
    with pytest.raises(ValueError):
        spec.to_unit(5.0 + 1e-6)


# ---- N = 2 (ei_2pop) ---------------------------------------------------------------------
# A symmetric W and equal taus would make a transposed Jacobian or a swapped 1/tau row-scaling
# invisible. These tests therefore use a W that is NOT symmetric and unequal taus, and assert
# that this is really the case.

EI_PARAMS = {"w_EE": 7.0, "w_EI": 3.0, "w_IE": 11.0, "w_II": 2.0, "I_E": 1.5, "I_I": 4.0}


def test_ei_structure_and_dale_signs(ei):
    net = ei.net(EI_PARAMS)
    np.testing.assert_array_equal(net.W, [[7.0, -3.0], [11.0, -2.0]])
    assert (net.W[:, 0] > 0).all() and (net.W[:, 1] < 0).all()  # E column +, I column -
    assert not np.allclose(net.W, net.W.T)  # non-symmetric
    assert net.tau[0] != net.tau[1]  # unequal taus


def test_ei_derivative_and_jacobian_match_finite_differences(ei):
    rng = np.random.default_rng(10)
    h = 1e-6
    for _ in range(10):
        net = ei.net(_random_params(ei, rng))
        u = rng.uniform(-5, 15, size=2)
        fd = (rate_fn(u + h, net) - rate_fn(u - h, net)) / (2 * h)
        np.testing.assert_allclose(rate_deriv(u, net), fd, rtol=1e-6, atol=1e-9)
        r = rng.uniform(0, 1, size=2)
        fd_j = np.empty((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            fd_j[:, j] = (rhs(r + e, net) - rhs(r - e, net)) / (2 * h)
        np.testing.assert_allclose(jacobian(r, net), fd_j, rtol=1e-5, atol=1e-9)
        # J = diag(1/tau) @ residual_jacobian; unequal taus make this a row-scaling check
        np.testing.assert_allclose(
            jacobian(r, net), np.diag(1.0 / net.tau) @ residual_jacobian(r, net), rtol=1e-12
        )


def test_rhs_batch_equals_single_bitwise(ei):
    """Batched rows must go through the same arithmetic as lone states (simulator relies on it)."""
    rng = np.random.default_rng(11)
    net = ei.net(EI_PARAMS)
    R = rng.uniform(0, 1, size=(7, 2))
    batch = rhs(R, net)
    for i in range(7):
        np.testing.assert_array_equal(batch[i], rhs(R[i], net))
