"""Test 5: the expert objective is graded and ranks fits correctly; the naive one can be fooled."""

from dataclasses import replace

import numpy as np
import pytest

from bench.checker import check
from bench.config import BudgetConfig
from bench.env import Environment
from bench.scoring import expert_v2_objective, expert_v2_terms

# ---- expert objective -------------------------------------------------------------------


@pytest.mark.parametrize("which", ["toy", "ei"])
def test_expert_separates_success_from_failure_and_counts_unmatched(
    which, cfg_toy, cfg_ei, task_toy, task_ei
):
    """By construction: all matched -> value <= tol; each unmatched target adds >= penalty/K.

    So a better fit (more targets matched, smaller errors) always scores lower.
    """
    cfg, task = (cfg_toy, task_toy) if which == "toy" else (cfg_ei, task_ei)
    env = Environment.from_task(cfg, task)
    model, K, pen = cfg.model(), cfg.task.K, cfg.scoring.expert_penalty
    rng = np.random.default_rng(5)
    u0 = env.space.to_unit(task.planted_params)
    seen_unmatched = set()
    for i in range(150):
        # Perturb the planted solution at growing distances (in unit space) so that full,
        # partial and no matches all occur; i == 0 is the exact planted solution.
        radius = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0][i % 6]
        params = env.space.from_unit(np.clip(u0 + radius * rng.normal(size=u0.shape), 0, 1))
        res = check(model, params, task.targets, cfg.checker)
        value = env.evaluate(params, "expert_v1")
        n_unmatched = res.target_matched.count(False)
        seen_unmatched.add(n_unmatched)
        if n_unmatched == 0:
            assert value <= cfg.task.tol
        else:
            assert value >= n_unmatched * pen / K
    assert 0 in seen_unmatched and len(seen_unmatched) >= 2  # both successes and failures


def test_expert_is_zero_at_the_planted_solution(cfg_toy, task_toy):
    env = Environment.from_task(cfg_toy, task_toy)
    assert env.evaluate(task_toy.planted_params, "expert_v1") == pytest.approx(0.0, abs=1e-9)


def test_expert_is_never_constant_across_a_monostable_bistable_sweep(cfg_toy, task_toy):
    """Sweep the recurrent weight w around the planted value: monostable below, bistable above.

    (The planted toy sits close to the edge of bistability, so the sweep is narrow.)"""
    env = Environment.from_task(cfg_toy, task_toy)
    res_by_w, values = [], []
    for w in np.linspace(2.0, 6.0, 60):
        params = {**task_toy.planted_params, "w": float(w)}
        res_by_w.append(check(cfg_toy.model(), params, task_toy.targets, cfg_toy.checker).n_stable)
        values.append(env.evaluate(params, "expert_v1"))
    assert 1 in res_by_w and 2 in res_by_w  # the sweep really crosses mono- and bistability
    assert len(set(values)) == len(values)  # no two points tie, so no flat stretch anywhere
    assert max(values) - min(values) > 0.1


# ---- naive objective --------------------------------------------------------------------


def test_naive_objective_can_be_fooled_by_a_slow_transient_past_a_fold(cfg_toy):
    """Documented failure of the naive objective (found, not forced).

    Toy with w=10, k=1: the low and middle fixed points merge in a saddle-node at
    I* ~ 1.8095. Just beyond it (I = I* + 1e-3) the system is MONOSTABLE: the checker fails it
    and the expert objective is large. But the dynamics are still extremely slow near the
    vanished low state (a "ghost"), so a simulation started low is still sitting at the old
    low state when time T runs out. The naive objective sees that and reports a near-perfect
    fit, better than a genuinely successful nearby solution.
    """
    w = 10.0
    s = (1 - np.sqrt(1 - 4 / w)) / 2
    I_star = 5 - w * s + np.log(s / (1 - s))
    targets = np.array([[s], [0.99]])  # low target at the ghost location, high target on top
    env = Environment(cfg_toy, targets, seed=0)

    ghost = {"w": w, "I": float(I_star + 1e-3), "k": 1.0}  # monostable
    real = {"w": w, "I": float(I_star - 1e-2), "k": 1.0}  # truly bistable, targets realized
    assert env.final_report(real).success and not env.final_report(ghost).success

    naive_ghost, naive_real = env.evaluate(ghost, "naive"), env.evaluate(real, "naive")
    assert naive_ghost < naive_real  # fooled: the failing set looks BETTER to the naive score
    assert env.evaluate(ghost, "expert_v1") > 0.9  # the expert objective is not fooled
    assert env.evaluate(real, "expert_v1") < 0.02


# ---- expert_v2 (definition pre-registered in NOTES.md) -----------------------------------------


@pytest.mark.parametrize("which", ["toy", "ei"])
def test_expert_v2_is_zero_when_the_targets_are_stable_fixed_points(
    which, cfg_toy, cfg_ei, task_toy, task_ei
):
    """Noise-free task + planted parameters: every target IS a stable fixed point, so the score
    is 0 (up to the root-finding precision of the checker that produced the targets)."""
    cfg, task = (cfg_toy, task_toy) if which == "toy" else (cfg_ei, task_ei)
    assert cfg.task.noise == 0.0
    env = Environment.from_task(cfg, task)
    assert env.evaluate(task.planted_params, "expert_v2") == pytest.approx(0.0, abs=1e-8)


def test_expert_v2_is_positive_when_a_target_is_the_unstable_fixed_point(toy, cfg_toy):
    """Hand-checked toy (w=10, I=0, k=1): fixed points 0.0072 (stable), 0.5 (UNSTABLE, J = +0.15
    per time unit with tau = 10) and 0.9928 (stable). A target at 0.5 costs 0.15 * 10 + delta."""
    params = {"w": 10.0, "I": 0.0, "k": 1.0}
    targets = np.array([[0.5], [0.9928]])
    residual, stab = expert_v2_terms(toy, params, targets, cfg_toy.scoring.v2_delta)
    assert residual[0] == pytest.approx(0.0, abs=1e-12)  # 0.5 is exactly a fixed point...
    assert stab[0] == pytest.approx(0.15 * 10 + 0.01, abs=1e-9)  # ...but an unstable one
    assert stab[1] == 0.0 and residual[1] < 1e-3  # 0.9928 is stable (rounded value: tiny residual)
    score = expert_v2_objective(toy, params, targets, 0.01, 1.0)
    assert score == pytest.approx((1.51 + residual[1]) / 2, abs=1e-9) and score > 0.7


def test_expert_v2_has_no_jump_where_the_checker_changes_its_verdict(cfg_toy, task_toy):
    """Slide I until the checker flips from success to failure and compare the scores just on
    both sides: expert_v1 jumps (an unmatched target costs >= 1), expert_v2 does not."""
    cfg = replace(cfg_toy, budget=BudgetConfig(1000))
    env = Environment.from_task(cfg, task_toy)
    model, params = cfg.model(), dict(task_toy.planted_params)

    def ok(i):
        return check(model, {**params, "I": i}, task_toy.targets, cfg.checker).success

    grid = np.linspace(0.0, 10.0, 401)
    flips = [(a, b) for a, b in zip(grid[:-1], grid[1:], strict=True) if ok(a) != ok(b)]
    assert flips, "expected the verdict to change somewhere along I"
    lo, hi = flips[0]
    good_at_lo = ok(lo)
    for _ in range(45):  # bisect to the boundary
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) == good_at_lo else (lo, mid)
    side_a, side_b = {**params, "I": lo}, {**params, "I": hi}
    assert ok(lo) != ok(hi)
    jump_v1 = abs(env.evaluate(side_a, "expert_v1") - env.evaluate(side_b, "expert_v1"))
    jump_v2 = abs(env.evaluate(side_a, "expert_v2") - env.evaluate(side_b, "expert_v2"))
    assert jump_v1 > 0.4 and jump_v2 < 1e-6
