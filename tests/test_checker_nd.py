"""Checker tests for N>=2 (ei_2pop), including test 3 (checker vs simulator) for ei_2pop."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from bench.checker import CheckerConfig, check
from bench.models.instances import model_from_dict
from bench.simulate import simulate_batch

# Hand-checked case. With w_IE = w_II = 0 and I_I = theta_I = 5 the I population sits at
# exactly 0.5 whatever E does. E then receives 10 * r_E - 2 * 0.5 + 1 = 10 * r_E: identical to
# the bistable toy (w=10, I=0, theta=5, k=1). Fixed points: E ~ 0.0072, 0.5, ~0.9928, I = 0.5.
DECOUPLED = {"w_EE": 10.0, "w_EI": 2.0, "w_IE": 0.0, "w_II": 0.0, "I_E": 1.0, "I_I": 5.0}


def test_decoupled_ei_hand_checked(ei):
    res = check(ei, DECOUPLED, np.array([[0.0], [1.0]]))  # targets on E only
    assert len(res.fixed_points) == 3
    e = [fp.r[0] for fp in res.fixed_points]
    assert e[0] == pytest.approx(0.0072, abs=1e-3)
    assert e[1] == pytest.approx(0.5, abs=1e-9)
    assert e[2] == pytest.approx(0.9928, abs=1e-3)
    assert all(fp.r[1] == pytest.approx(0.5, abs=1e-9) for fp in res.fixed_points)
    assert [fp.stable for fp in res.fixed_points] == [True, False, True]
    assert res.success and res.n_stable == 2
    assert res.global_search_exhaustive is False  # never claimed for N>=2
    assert res.warnings == []
    # eigenvalues: E as in the toy (+-), I decays at -1/tau_I = -0.2
    assert min(res.fixed_points[0].eig_real) == pytest.approx(-0.2, abs=1e-9)


def test_local_search_finds_targets_without_the_global_search(ei):
    """With n_starts=1 the global part is nearly useless; the target-seeded Newton must still
    find the stable states next to the targets (this is what decides success)."""
    cfg = CheckerConfig(n_starts=1)
    res = check(ei, DECOUPLED, np.array([[0.0], [1.0]]), cfg)
    assert res.success


def test_partial_vs_full_observation(ei):
    """Same model with both populations observed: targets are (r_E, r_I) pairs."""
    spec = yaml.safe_load((Path(__file__).parents[1] / "configs" / "ei_2pop.yaml").read_text())
    spec["observed"] = ["E", "I"]
    both = model_from_dict(spec)
    res = check(both, DECOUPLED, np.array([[0.0072, 0.5], [0.9928, 0.5]]))
    assert res.success
    wrong_i = check(both, DECOUPLED, np.array([[0.0072, 0.9], [0.9928, 0.5]]))
    assert not wrong_i.success  # I is constrained now, and 0.9 is not reachable


def test_result_depends_only_on_params_and_targets(ei):
    """Same inputs -> identical result, no matter what was evaluated in between."""
    targets = np.array([[0.0], [1.0]])
    a = check(ei, DECOUPLED, targets)
    other = {**DECOUPLED, "w_EE": 15.0, "w_IE": 6.0}
    check(ei, other, targets)
    b = check(ei, DECOUPLED, targets)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())


def test_different_seed_same_roots_for_a_simple_case(ei):
    """Sanity: the global search seed changes the sample, but all roots of this easy case are
    found either way (so the seed is not hiding anything here)."""
    t = np.array([[0.0], [1.0]])
    a = check(ei, DECOUPLED, t, CheckerConfig(seed=0))
    b = check(ei, DECOUPLED, t, CheckerConfig(seed=1))
    np.testing.assert_allclose([fp.r for fp in a.fixed_points], [fp.r for fp in b.fixed_points])


# ---- Test 3: checker vs simulator, ei_2pop ----------------------------------------------


def test_checker_and_simulator_agree_on_ei(ei):
    """Simulate from many seeded-random ICs for random parameters and compare to the checker.

    Two kinds of mismatch are possible; both are reported with parameters and numbers:
      A. a simulation SETTLES on a state that is not one of the checker's stable fixed points
         -> the checker's search missed a stable state (strict failure).
      B. a stable state the checker found is reached by no simulation. This is not
         necessarily an error (its basin may be tiny), so it is reported as a warning.
    Simulations that do not settle (limit cycles are possible in E/I networks) are counted and
    must coincide with the checker finding no stable state... they are not otherwise compared.
    """
    rng = np.random.default_rng(2024)
    n_params, n_ic = 80, 80
    type_a: list[str] = []
    type_b: list[str] = []
    n_unsettled = 0
    n_bistable = 0
    for _ in range(n_params):
        params = ei.space.from_unit(rng.random(ei.space.dim))
        res = check(ei, params, np.array([[0.5]]))
        stable = np.array([res.fixed_points[i].r for i in res.stable_indices]).reshape(-1, 2)
        n_bistable += res.n_stable >= 2
        reached: set[int] = set()
        for sim in simulate_batch(ei, params, rng.random((n_ic, 2)), T=3000.0):
            if not sim.settled:
                n_unsettled += 1
                continue
            if len(stable) == 0:
                type_a.append(f"{params}: sim settled at {sim.final_state}, checker: no stable")
                continue
            dist = np.linalg.norm(stable - np.array(sim.final_state), axis=1)
            if dist.min() > 1e-3:
                type_a.append(f"{params}: sim settled at {sim.final_state}, checker: {stable}")
            else:
                reached.add(int(dist.argmin()))
        for j in set(range(len(stable))) - reached:
            type_b.append(f"{params}: stable state {stable[j]} reached by no simulation")

    assert not type_a, "checker MISSED stable states found by simulation:\n" + "\n".join(type_a)
    if type_b:
        warnings.warn(
            "stable states never reached by simulation (tiny basin or checker false positive):\n"
            + "\n".join(type_b),
            stacklevel=1,
        )
    assert n_bistable >= 8, "sweep did not exercise multistable parameters"
    print(f"\nei cross-check: {n_bistable}/{n_params} multistable, {n_unsettled} unsettled sims")


# ---- Limit-cycle regression ---------------------------------------------------------------

# E/I circuit with slow inhibition (tau_I = 20) whose only fixed point is an unstable focus:
# the network oscillates forever. This is a config change of the same model, not a new instance.
LIMIT_CYCLE = {
    "w_EE": 16.537,
    "w_EI": 17.71,
    "w_IE": 13.207,
    "w_II": 4.911,
    "I_E": 7.685,
    "I_I": 2.117,
}


@pytest.fixture(scope="module")
def ei_slow_inhibition():
    spec = yaml.safe_load((Path(__file__).parents[1] / "configs" / "ei_2pop.yaml").read_text())
    spec["fixed"]["tau_I"] = 20.0
    return model_from_dict(spec)


def test_limit_cycle_has_no_stable_state_and_does_not_settle(ei_slow_inhibition):
    """Regression: a parameter set with fixed points but NO stable one must be handled.

    (It originally crashed the checker: reshaping an empty list of stable states.)
    """
    m = ei_slow_inhibition
    res = check(m, LIMIT_CYCLE, np.array([[0.2], [0.8]]))
    assert len(res.fixed_points) == 1
    fp = res.fixed_points[0]
    assert fp.r == pytest.approx([0.470, 0.598], abs=1e-3)
    assert fp.eig_real == pytest.approx([0.10, 0.10], abs=5e-3)  # unstable focus ...
    assert sorted(np.abs(fp.eig_imag)) == pytest.approx([0.16, 0.16], abs=5e-3)  # ... spiralling
    assert res.n_stable == 0 and not res.success
    assert res.target_assignment == [None, None] and res.margin_eig is None

    # The simulator agrees: it never comes to rest, and E keeps swinging with large amplitude.
    sim = simulate_batch(m, LIMIT_CYCLE, np.array([[0.1, 0.1]]), T=1000.0)[0]
    assert not sim.settled
    x, e_trace = np.array(sim.final_state), []
    for _ in range(60):  # 60 chunks of 5 time units after the transient
        x = np.array(simulate_batch(m, LIMIT_CYCLE, x[None, :], T=5.0)[0].final_state)
        e_trace.append(x[0])
    assert max(e_trace) - min(e_trace) > 0.5
