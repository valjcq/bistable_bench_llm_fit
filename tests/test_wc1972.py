"""Wilson & Cowan (1972) model: generalized dynamics, bounds, the paper's claims, expert_v2.

Two labelled families (see docs/model_cards/wilson_cowan_1972.md for the tags):
  [paper]    qualitative claims of the paper itself (number and stability of steady states,
             limit cycle, hysteresis loops read by eye from the figures);
  [computed] regression on values WE recomputed from the paper's equations (the card's steady-state
             table and P intervals), for k = S(+inf) and k = 1.
The remaining tests check the machinery: Jacobian, state bounds, checker vs simulator.
"""

import importlib.util
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from bench.checker import check
from bench.models.instances import load_model
from bench.models.rate_network import (
    fixed_point_residual,
    jacobian,
    rate_deriv,
    rate_fn,
    residual_jacobian,
    rhs,
    sigmoid,
    state_bounds,
)
from bench.models.wc1972 import free_params, paper_sets, with_free_input
from bench.scoring import expert_v2_objective, expert_v2_terms
from bench.simulate import simulate_batch
from bench.tasks import target_window_bounds

ROOT = Path(__file__).resolve().parents[1]
TARGETS = np.array([[0.0], [0.4]])  # only seed the checker's local search; the model decides


@pytest.fixture(scope="module")
def wc():
    return with_free_input(load_model(ROOT / "configs/wc1972.yaml"))  # k = S(+inf)


@pytest.fixture(scope="module")
def wc_k1():
    return with_free_input(load_model(ROOT / "configs/wc1972_k1.yaml"))  # k = 1


def params_of(name: str, P: float | None = None) -> dict[str, float]:
    s = paper_sets()[name]
    return {**free_params(s), "P": s["P"] if P is None else P}


@cache
def _model(k_one: bool):
    cfg = "wc1972_k1" if k_one else "wc1972"
    return with_free_input(load_model(ROOT / f"configs/{cfg}.yaml"))


@cache
def n_stable_sweep(set_name: str, k_one: bool) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Number of stable states of a paper set along P = -0.80 ... 1.20 in steps of 0.01."""
    model = _model(k_one)
    Ps = tuple(float(p) for p in np.round(np.arange(-0.8, 1.2001, 0.01), 3))
    ns = tuple(check(model, params_of(set_name, P), TARGETS).n_stable for P in Ps)
    return Ps, ns


def intervals(set_name: str, k_one: bool, n: int) -> list[tuple[float, float]]:
    """Maximal P intervals (step 0.01) where the number of stable states is exactly n."""
    Ps, ns = n_stable_sweep(set_name, k_one)
    out: list[tuple[float, float]] = []
    for P, k in zip(Ps, ns, strict=True):
        if k == n:
            if out and abs(P - out[-1][1] - 0.01) < 1e-9:
                out[-1] = (out[-1][0], P)
            else:
                out.append((P, P))
    return out


# ---- the generalized dynamics ---------------------------------------------------------------


def test_default_k_is_the_sigmoid_maximum_and_matches_the_card(wc):
    net = wc.net(params_of("fig4_5"))
    np.testing.assert_allclose(net.r_max, [0.966, 0.982], atol=1e-3)  # card: k_e, k_i for Fig. 4
    assert net.rref.tolist() == [1.0, 1.0] and net.variant == "wc_shifted"


def test_shifted_sigmoid_is_zero_at_zero_and_has_the_papers_maximum_slope(wc):
    net = wc.net(params_of("fig4_5"))
    np.testing.assert_allclose(sigmoid(np.zeros(2), net), 0.0, atol=1e-15)  # eq. 15: S(0) = 0
    # eq. 16: max S'(x) = a / 4, reached at x = theta
    slope = rate_deriv(net.theta, net, np.zeros(2)) / net.r_max  # strip the amplitude (r = 0)
    np.testing.assert_allclose(slope, [1.2 / 4, 1.0 / 4])


def test_logistic_with_zero_refractory_term_is_the_plain_rate_network(toy, ei):
    """rref = 0 and the logistic variant reproduce  (-r + r_max expit((u - theta)/k)) / tau
    EXACTLY (bit for bit) on random states; tests/test_golden_logistic.py covers the checker,
    simulator and objectives end to end."""
    rng = np.random.default_rng(3)
    for model in (toy, ei):
        net = model.net(model.space.from_unit(rng.random(model.space.dim)))
        assert not net.rref.any() and not net.shift.any()
        for _ in range(20):
            r = rng.random(model.n_pops)
            u = (net.W * r[None, :]).sum(axis=-1) + net.I_ext
            old = (-r + net.r_max * expit((u - net.theta) / net.k)) / net.tau
            assert np.array_equal(rhs(r, net), old)


@pytest.mark.parametrize("set_name", ["fig4_5", "fig6", "fig7_8", "fig11"])
def test_jacobian_matches_finite_differences_and_the_cards_formulas(wc, set_name):
    """Jacobian vs central differences on the Wilson-Cowan parameters, and vs the four entries
    J11..J22 written in the model card (r_e = r_i = 1)."""
    p = params_of(set_name)
    net = wc.net(p)
    lo, hi = state_bounds(net)
    rng = np.random.default_rng(11)
    h = 1e-6
    for _ in range(10):
        r = lo + rng.random(2) * (hi - lo)
        fd = np.empty((2, 2))
        for j in range(2):
            d = np.zeros(2)
            d[j] = h
            fd[:, j] = (rhs(r + d, net) - rhs(r - d, net)) / (2 * h)
        np.testing.assert_allclose(jacobian(r, net), fd, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(residual_jacobian(r, net), jacobian(r, net) * net.tau[:, None])
        # the card's entries, from a, theta and the weights directly
        E, I_ = r
        xe, xi = p["c1"] * E - p["c2"] * I_ + p["P"], p["c3"] * E - p["c4"] * I_
        sig = lambda x, a, th: 1 / (1 + np.exp(-a * (x - th))) - 1 / (1 + np.exp(a * th))  # noqa: E731
        dsig = lambda x, a, th: a * (lambda s: s * (1 - s))(1 / (1 + np.exp(-a * (x - th))))  # noqa: E731
        ke, ki = net.r_max
        S_e, S_i = sig(xe, p["a_e"], p["theta_e"]), sig(xi, p["a_i"], p["theta_i"])
        dSe, dSi = dsig(xe, p["a_e"], p["theta_e"]), dsig(xi, p["a_i"], p["theta_i"])
        card = np.array(
            [
                [-1 - S_e + (ke - E) * p["c1"] * dSe, -(ke - E) * p["c2"] * dSe],
                [(ki - I_) * p["c3"] * dSi, -1 - S_i - (ki - I_) * p["c4"] * dSi],
            ]
        )
        np.testing.assert_allclose(jacobian(r, net) * 10.0, card, rtol=1e-12, atol=1e-12)


def test_rate_fn_without_state_is_refused_when_the_refractory_term_is_nonzero(wc):
    net = wc.net(params_of("fig4_5"))
    with pytest.raises(ValueError, match="state r is required"):
        rate_fn(np.zeros(2), net)
    rate_fn(np.zeros(2), net, np.zeros(2))


# ---- model-provided state bounds -----------------------------------------------------------


def test_bounds_are_the_old_box_for_the_logistic_models(toy, ei):
    for model in (toy, ei):
        net = model.net(model.space.from_unit(np.full(model.space.dim, 0.5)))
        lo, hi = state_bounds(net)
        assert np.array_equal(lo, np.zeros(model.n_pops)) and np.array_equal(hi, net.r_max)
        assert not np.signbit(lo).any()  # +0.0, never -0.0


def test_wc_bounds_follow_the_derivation(wc, wc_k1):
    """hi = k^2 / (1 + k) for k = S(+inf) (about 0.48), lo slightly negative; for k = 1 the
    bound uses the true asymptote S(+inf) < 1, so it is a bit below 1/2."""
    lo, hi = wc.state_bounds(params_of("fig4_5"))
    net = wc.net(params_of("fig4_5"))
    np.testing.assert_allclose(hi, net.r_max**2 / (1 + net.r_max), rtol=1e-12)
    assert 0.47 < hi[0] < 0.49 and 0.48 < hi[1] < 0.50 and (lo < 0).all() and (lo > -0.1).all()
    _, hi1 = wc_k1.state_bounds(params_of("fig4_5"))
    s_max = 1 - wc.net(params_of("fig4_5")).shift
    np.testing.assert_allclose(hi1, s_max / (1 + s_max), rtol=1e-12)
    assert (hi1 < 0.5).all()


@pytest.mark.parametrize("k_one", [False, True])
@pytest.mark.parametrize("set_name", ["fig4_5", "fig6", "fig7_8", "fig11"])
def test_box_is_invariant_and_contains_every_fixed_point(set_name, k_one):
    """The flow points inward on the faces of the box, trajectories stay inside, fixed points lie
    inside (the derivation of `state_bounds` in executable form)."""
    model = _model(k_one)
    p = params_of(set_name)
    net = model.net(p)
    lo, hi = state_bounds(net)
    rng = np.random.default_rng(5)
    for _ in range(200):
        other = rng.random() * (hi - lo) + lo
        for i in range(2):
            top, bottom = other.copy(), other.copy()
            top[i], bottom[i] = hi[i], lo[i]
            assert rhs(top, net)[i] <= 1e-12 and rhs(bottom, net)[i] >= -1e-12
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    for sim in simulate_batch(model, p, corners, 400.0):
        assert (np.array(sim.final_state) >= lo - 1e-9).all()
        assert (np.array(sim.final_state) <= hi + 1e-9).all()
    for fp in check(model, p, TARGETS).fixed_points:
        assert (np.array(fp.r) >= lo - 1e-9).all() and (np.array(fp.r) <= hi + 1e-9).all()


def test_planting_window_is_a_fraction_of_the_reference_range(wc, toy):
    """5% to 95% of the model's configured reference range: [0, 0.5] for Wilson-Cowan (so
    0.025 to 0.475), [0, 1] for the logistic models (0.05 to 0.95, exactly as before)."""
    lo, hi = target_window_bounds(wc, (0.05, 0.95))
    assert lo.tolist() == [0.025] and hi.tolist() == [0.475]
    lo_t, hi_t = target_window_bounds(toy, (0.05, 0.95))
    assert lo_t[0] == 0.05 * 1.0 and hi_t[0] == 0.95 * 1.0


# ---- [paper] qualitative claims -------------------------------------------------------------


@pytest.mark.paper
def test_paper_fig4_has_three_steady_states_two_stable(wc):
    res = check(wc, params_of("fig4_5"), TARGETS)
    assert len(res.fixed_points) == 3 and res.n_stable == 2


@pytest.mark.paper
def test_paper_fig6_has_two_stable_states_at_p0(wc):
    assert check(wc, params_of("fig6"), TARGETS).n_stable == 2


@pytest.mark.paper
def test_paper_fig78_has_three_stable_and_two_unstable_states_at_p0(wc):
    res = check(wc, params_of("fig7_8"), TARGETS)
    assert res.n_stable == 3 and len(res.fixed_points) - res.n_stable == 2


@pytest.mark.paper
def test_paper_fig11_single_unstable_focus_and_the_simulation_oscillates(wc):
    p = params_of("fig11")
    res = check(wc, p, TARGETS)
    assert len(res.fixed_points) == 1 and res.n_stable == 0
    fp = res.fixed_points[0]
    assert not fp.stable and max(abs(x) for x in fp.eig_imag) > 0  # a focus, not a node
    x, E = np.array([0.1, 0.05]), []
    for _ in range(400):  # 2000 time units in chunks; record E over the last ones
        sim = simulate_batch(wc, p, x[None], 5.0)[0]
        x = np.array(sim.final_state)
        E.append(x[0])
    assert not sim.settled
    assert max(E[-100:]) - min(E[-100:]) > 0.05  # sustained oscillation (observed amplitude 0.16)


@pytest.mark.paper
def test_paper_fig5_hysteresis_edges_within_0p1_of_the_cards_by_eye_reading():
    """Card: the Fig. 5 loop is roughly between P = -0.35 and P = 0.3 (by eye, k = S(+inf))."""
    loop = intervals("fig4_5", False, 2)
    assert len(loop) == 1
    assert abs(loop[0][0] - (-0.35)) <= 0.1 and abs(loop[0][1] - 0.3) <= 0.1


@pytest.mark.paper
def test_paper_fig6_hysteresis_edges_within_0p1_of_the_cards_by_eye_reading():
    """Card: first loop about P = -0.2 ... 0.15, second about 0.4 ... 0.9 (by eye)."""
    loops = intervals("fig6", False, 2)
    assert len(loops) == 2
    for (a, b), (ea, eb) in zip(loops, [(-0.2, 0.15), (0.4, 0.9)], strict=True):
        assert abs(a - ea) <= 0.1 and abs(b - eb) <= 0.1


# ---- [computed] regression on our recomputed values -----------------------------------------

# The card's "Recomputed steady states" table (k = S(+inf), Q = 0, tau_e = tau_i):
# (E, I, stable, is_focus). is_focus is None where the card does not say.
CARD_TABLE = {
    "fig4_5": (0.0, [(0.0, 0.0, True), (0.190, 0.068, False), (0.440, 0.226, True)]),
    "fig6": (0.0, [(0.0, 0.0, True), (0.095, 0.000, False), (0.178, 0.116, True)]),
    "fig7_8": (
        0.0,
        [(0.0, 0.0, True), (0.095, 0.0, False), (0.204, 0.189, True),
         (0.380, 0.500, False), (0.454, 0.500, True)],
    ),
    "fig11": (1.25, [(0.202, 0.107, False)]),
}  # fmt: skip


@pytest.mark.parametrize("set_name", list(CARD_TABLE))
@pytest.mark.computed
def test_computed_steady_state_table_k_splus_inf(wc, set_name):
    P, expected = CARD_TABLE[set_name]
    fps = check(wc, params_of(set_name, P), TARGETS).fixed_points
    assert len(fps) == len(expected)
    for fp, (E, I_, stable) in zip(fps, expected, strict=True):
        assert fp.r == pytest.approx([E, I_], abs=2e-3)
        assert fp.stable == stable


def _load_wc_check():
    spec = importlib.util.spec_from_file_location("wc_check", ROOT / "scripts/wc_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("k_one", [False, True])
@pytest.mark.parametrize("set_name", ["fig4_5", "fig6", "fig7_8", "fig11"])
@pytest.mark.computed
def test_computed_checker_agrees_with_the_independent_wc_check_script(set_name, k_one):
    """scripts/wc_check.py reduces eq. 12 = 0 to I(E) and scans E (an independent method).
    The checker must find the same steady states, with the same stability, for both k choices."""
    wc_check = _load_wc_check()
    s = paper_sets()[set_name]
    card = {  # the script uses the card's short names
        "c1": s["c1"], "c2": s["c2"], "c3": s["c3"], "c4": s["c4"],
        "ae": s["a_e"], "te": s["theta_e"], "ai": s["a_i"], "ti": s["theta_i"],
    }  # fmt: skip
    P = s["P"]
    ref = wc_check.fixed_points(card, P=P, kmode="one" if k_one else "max")
    got = check(_model(k_one), params_of(set_name, P), TARGETS).fixed_points
    assert len(got) == len(ref)
    for fp, (E, I_, stable, _) in zip(got, sorted(ref, key=lambda t: t[0]), strict=True):
        assert fp.r == pytest.approx([E, I_], abs=1e-6)
        assert fp.stable == stable


# The card's "P intervals with >= 2 stable states" (Q = 0, step 0.01), as (n_stable, from, to).
CARD_INTERVALS = {
    False: {  # k = S(+inf)
        "fig4_5": [(2, -0.39, 0.30)],
        "fig6": [(2, -0.20, 0.15), (2, 0.42, 0.94)],
        "fig7_8": [(2, -0.35, -0.14), (3, -0.13, 0.21), (2, 0.22, 0.74)],
    },
    True: {  # k = 1
        "fig4_5": [(2, -0.51, 0.28)],
        "fig6": [(2, -0.24, 0.13), (2, 0.21, 0.87)],
        "fig7_8": [(2, -0.36, -0.27), (3, -0.26, 0.20), (2, 0.21, 0.71)],
    },
}


@pytest.mark.parametrize("k_one", [False, True])
@pytest.mark.parametrize("set_name", ["fig4_5", "fig6", "fig7_8"])
@pytest.mark.computed
def test_computed_p_intervals_with_two_or_more_stable_states(set_name, k_one):
    expected = CARD_INTERVALS[k_one][set_name]
    for n, a, b in expected:
        got = intervals(set_name, k_one, n)
        assert any(abs(x - a) < 1e-9 and abs(y - b) < 1e-9 for x, y in got), (n, a, b, got)
    # and nothing else has two or more stable states
    Ps, ns = n_stable_sweep(set_name, k_one)
    covered = {round(p, 2) for _, a, b in expected for p in np.arange(a, b + 0.005, 0.01)}
    assert all(k < 2 or round(P, 2) in covered for P, k in zip(Ps, ns, strict=True))


# ---- checker vs simulator on random Wilson-Cowan parameters ---------------------------------


def test_checker_and_simulator_agree_on_random_wc_parameters(wc):
    """Random parameters in the search box of configs/wc1972.yaml (c in [0.5, 25], a in [1, 7],
    theta in [2, 5]), P = Q = 0. Simulate from random initial states inside the state bounds.
      A. a simulation SETTLES on a state that is not one of the checker's stable fixed points
         -> the checker missed a stable state (strict failure);
      B. a stable state the checker found that no simulation reaches -> warning only (its basin
         may be small).
    """
    rng = np.random.default_rng(2025)
    n_params, n_ic = 40, 40
    type_a, type_b = [], []
    n_multistable = n_unsettled = 0
    for _ in range(n_params):
        p = wc.space.from_unit(rng.random(wc.space.dim))
        p = {**p, "P": 0.0}
        res = check(wc, p, np.array([[0.2]]))
        stable = np.array([res.fixed_points[i].r for i in res.stable_indices]).reshape(-1, 2)
        n_multistable += res.n_stable >= 2
        lo, hi = wc.state_bounds(p)
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        # random states plus the box corners: saturated states have small basins that random
        # states can miss, the corners reach them
        x0s = np.vstack([lo + rng.random((n_ic, 2)) * (hi - lo), corners])
        reached: set[int] = set()
        for sim in simulate_batch(wc, p, x0s, T=2000.0):
            if not sim.settled:
                n_unsettled += 1
                continue
            dist = (
                np.linalg.norm(stable - np.array(sim.final_state), axis=1)
                if len(stable)
                else np.array([np.inf])
            )
            if dist.min() > 1e-3:
                type_a.append(f"{p}: sim settled at {sim.final_state}, checker stable: {stable}")
            else:
                reached.add(int(dist.argmin()))
        type_b += [
            f"{p}: stable {stable[j]} never reached" for j in set(range(len(stable))) - reached
        ]
    assert not type_a, "checker MISSED stable states found by simulation:\n" + "\n".join(type_a)
    if type_b:
        warnings.warn("stable states reached by no simulation:\n" + "\n".join(type_b), stacklevel=1)
    assert n_multistable >= 5, "the sweep did not exercise multistable parameters"


# ---- expert_v2 on Wilson-Cowan --------------------------------------------------------------


def test_expert_v2_is_zero_at_a_paper_set_with_targets_from_its_own_stable_states(wc):
    """E observed, I unobserved (solved with brentq inside the model's state bounds)."""
    p = params_of("fig4_5")
    res = check(wc, p, TARGETS)
    targets = np.array([[fp.observed[0]] for fp in res.fixed_points if fp.stable])
    assert targets.shape == (2, 1)
    residual, stab = expert_v2_terms(wc, p, targets, 0.01)
    assert residual.max() < 1e-8 and stab.max() == 0.0
    assert expert_v2_objective(wc, p, targets, 0.01, 1.0) < 1e-8


def test_expert_v2_penalizes_a_wc_target_on_the_saddle(wc):
    p = params_of("fig4_5")
    saddle = [fp for fp in check(wc, p, TARGETS).fixed_points if not fp.stable][0]
    targets = np.array([[saddle.observed[0]], [0.4398]])
    residual, stab = expert_v2_terms(wc, p, targets, 0.01)
    assert residual[0] < 1e-8 and stab[0] == pytest.approx(saddle.max_re * 10.0 + 0.01, rel=1e-6)
    assert expert_v2_objective(wc, p, targets, 0.01, 1.0) > 0.3


def test_the_fixed_point_residual_vanishes_at_every_checker_root(wc):
    for name in ("fig4_5", "fig6", "fig7_8", "fig11"):
        p = params_of(name)
        net = wc.net(p)
        for fp in check(wc, p, TARGETS).fixed_points:
            assert np.abs(fixed_point_residual(np.array(fp.r), net)).max() < 1e-9
