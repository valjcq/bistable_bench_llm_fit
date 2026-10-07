"""Checker tests for N=1, and test 3 (checker vs simulator cross-check) for the toy."""

import json

import numpy as np
import pytest

from bench.checker import CheckerConfig, _match, check
from bench.simulate import SimConfig, simulate, simulate_batch

BISTABLE = {"w": 10.0, "I": 0.0, "k": 1.0}  # theta = 5 fixed: roots ~0.0072, 0.5, ~0.9928
MONOSTABLE = {"w": 2.0, "I": 1.0, "k": 1.0}


def test_bistable_hand_checked(toy):
    res = check(toy, BISTABLE, np.array([[0.0], [1.0]]))
    assert len(res.fixed_points) == 3
    # The default grid has 20001 nodes, so r = 0.5 is a node and g(0.5) == 0 exactly there.
    # It must be reported once, not twice (exact-zero node + the two neighbouring intervals).
    assert res.fixed_points[1].r[0] == pytest.approx(0.5, abs=1e-12)
    assert [fp.stable for fp in res.fixed_points] == [True, False, True]
    assert res.n_stable == 2 and res.global_search_exhaustive
    assert res.success and res.n_extra_stable == 0
    assert res.margin_eig > 0 and res.margin_dist == pytest.approx(0.5 - 0.0072, abs=1e-3)


def test_targets_on_unstable_point_fail(toy):
    res = check(toy, BISTABLE, np.array([[0.5], [0.99]]))
    assert not res.success  # 0.5 is a fixed point but unstable: it cannot be a target state


def test_monostable_cannot_realize_two_targets(toy):
    res = check(toy, MONOSTABLE, np.array([[0.2], [0.8]]))
    assert res.n_stable == 1 and not res.success
    assert res.target_assignment.count(None) == 1  # only one target gets a stable state
    assert res.margin_dist is None  # no unstable fixed point exists: undefined, not infinite


def test_extra_stable_states_flag(toy):
    one_target = np.array([[0.0]])
    res = check(toy, BISTABLE, one_target)
    assert res.success and res.n_extra_stable == 1  # reported, not a failure by default
    strict = check(toy, BISTABLE, one_target, CheckerConfig(forbid_extra_stable_states=True))
    assert not strict.success


def test_near_tangent_pair_is_found(toy):
    """Saddle-node pair closer than one grid cell (5e-5): invisible to a sign-change scan."""
    w = 10.0
    s = (1 - np.sqrt(1 - 4 / w)) / 2  # tangency needs w * s * (1 - s) = 1 with r = s
    I_star = 5 - w * s + np.log(s / (1 - s))  # then w r + I - theta = logit(s)
    just_below = {"w": w, "I": float(I_star - 1e-11), "k": 1.0}
    res = check(toy, just_below, np.array([[0.9]]))
    r = [fp.r[0] for fp in res.fixed_points]
    assert len(r) == 3 and r[1] - r[0] < 5e-5  # the close pair was resolved
    above = check(toy, {"w": w, "I": float(I_star + 1e-3), "k": 1.0}, np.array([[0.9]]))
    assert len(above.fixed_points) == 1  # past the bifurcation the pair is gone


def test_match_prefers_assignment_within_tolerance():
    # Plain min-sum picks (t0->s1, t1->s0) = 0.0 + 0.03; the valid one is (0.02 + 0.02).
    errors = np.array([[0.02, 0.0], [0.03, 0.02]])
    _, _, matched = _match(errors, tol=0.025)
    assert all(matched)


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e3])  # penalty is derived, so any scale works
def test_match_reports_real_errors_not_penalized_cost(scale):
    errors = scale * np.array([[0.02, 0.0], [0.03, 0.02]])
    assignment, errs, _ = _match(errors, tol=0.025 * scale)
    assert assignment == [0, 1]
    assert errs == [errors[0, 0], errors[1, 1]]  # exactly the entries, no penalty added


def test_out_of_tolerance_assignment_is_reported_and_extra_means_unassigned(toy):
    """One target at 0.5 (the unstable point): it is still ASSIGNED a stable state, with its
    real (large) error; the other stable state is then 'extra' = assigned to no target."""
    res = check(toy, BISTABLE, np.array([[0.5]]))
    assert not res.success and res.target_matched == [False]
    assert res.target_assignment[0] in res.stable_indices
    assert res.target_errors[0] == pytest.approx(0.5 - 0.0072, abs=1e-3)  # real error
    assert res.n_extra_stable == 1  # 2 stable states, only 1 assigned
    assert res.target_assignment[0] not in res.extra_stable_indices


def test_n1_bistable_always_has_an_unstable_point_between(toy):
    """Two stable points in 1D are always separated by an unstable one, so margin_dist must
    never be None when we ask the checker to match all its own stable states."""
    rng = np.random.default_rng(7)
    n_bistable = 0
    for _ in range(300):
        params = toy.space.from_unit(rng.random(toy.space.dim))
        res = check(toy, params, np.array([[0.5]]))
        if res.n_stable < 2:
            continue
        n_bistable += 1
        own = np.array([res.fixed_points[i].observed for i in res.stable_indices])
        full = check(toy, params, own)
        assert full.success and full.margin_dist is not None, params
        assert full.warnings == []
    assert n_bistable > 20  # the sweep really exercised the bistable case


def test_result_is_json_serializable_and_deterministic(toy):
    targets = np.array([[0.0], [1.0]])
    a, b = check(toy, BISTABLE, targets), check(toy, BISTABLE, targets)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())


def test_bad_target_shape_raises(toy):
    with pytest.raises(ValueError):
        check(toy, BISTABLE, np.array([0.1, 0.9]))  # must be 2D (K, n_obs)


# ---- Test 3: checker vs simulator, toy --------------------------------------------------


def test_simulator_reaches_known_fixed_point(toy):
    res = simulate(toy, BISTABLE, np.array([0.9]), T=2000.0)
    assert res.settled and res.final_state[0] == pytest.approx(0.9928, abs=1e-3)


def test_simulation_is_bit_reproducible(toy):
    a = simulate(toy, BISTABLE, np.array([0.3]), T=200.0)
    b = simulate(toy, BISTABLE, np.array([0.3]), T=200.0)
    assert a == b


def test_batch_result_is_independent_of_the_other_rows(toy):
    """Each trajectory in a batch is bit-identical to simulating that IC alone."""
    x0s = np.array([[0.02], [0.45], [0.55], [0.97]])
    batch = simulate_batch(toy, BISTABLE, x0s, T=2000.0)
    for x0, res in zip(x0s, batch, strict=True):
        assert res == simulate(toy, BISTABLE, x0, T=2000.0)


def test_settle_window_scales_with_tau(toy):
    """The rest window is hold_tau_factor * max(tau): going from 5 to 10 adds 5 * tau of time."""
    tau = toy.net(BISTABLE).tau.max()  # 10
    a = simulate(toy, BISTABLE, np.array([0.9]), 2000.0, SimConfig(hold_tau_factor=5.0))
    b = simulate(toy, BISTABLE, np.array([0.9]), 2000.0, SimConfig(hold_tau_factor=10.0))
    assert a.settled and b.settled and a.time_to_settle == b.time_to_settle
    assert b.t_end - a.t_end == pytest.approx(5.0 * tau, abs=0.2)


def test_checker_and_simulator_agree_on_toy(toy):
    """Strict for N=1: sims end on checker-stable states, and every stable state is reached.

    ICs are cell midpoints of a 100-cell grid over [0, r_max]: they never sit exactly on 0,
    r_max or a fixed point. In 1D every IC below the (single) unstable point flows to the
    lower stable state and every IC above it to the upper one, and there are at most 2 stable
    states (g'' changes sign once, so <= 3 roots), so these ICs reach every stable state
    unless a basin is narrower than one cell.
    """
    rng = np.random.default_rng(123)
    starts = ((np.arange(100) + 0.5) / 100)[:, None]
    for _ in range(25):
        params = toy.space.from_unit(rng.random(toy.space.dim))
        res = check(toy, params, np.array([[0.5]]))
        stable = np.array([res.fixed_points[i].r[0] for i in res.stable_indices])
        reached = set()
        for sim in simulate_batch(toy, params, starts, T=5000.0):
            assert sim.settled, (params, sim)
            dist = np.abs(stable - sim.final_state[0])
            assert dist.min() < 1e-3, (params, sim.final_state, stable)
            reached.add(int(dist.argmin()))
        assert reached == set(range(len(stable))), (params, stable, reached)
