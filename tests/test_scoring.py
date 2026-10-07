"""Test 5: the expert objective is graded and ranks fits correctly; the naive one can be fooled."""

import numpy as np
import pytest

from bench.checker import check
from bench.env import Environment

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
        value = env.evaluate(params, "expert")
        n_unmatched = res.target_matched.count(False)
        seen_unmatched.add(n_unmatched)
        if n_unmatched == 0:
            assert value <= cfg.task.tol
        else:
            assert value >= n_unmatched * pen / K
    assert 0 in seen_unmatched and len(seen_unmatched) >= 2  # both successes and failures


def test_expert_is_zero_at_the_planted_solution(cfg_toy, task_toy):
    env = Environment.from_task(cfg_toy, task_toy)
    assert env.evaluate(task_toy.planted_params, "expert") == pytest.approx(0.0, abs=1e-9)


def test_expert_is_never_constant_across_a_monostable_bistable_sweep(cfg_toy, task_toy):
    """Sweep the recurrent weight w around the planted value: monostable below, bistable above.

    (The planted toy sits close to the edge of bistability, so the sweep is narrow.)"""
    env = Environment.from_task(cfg_toy, task_toy)
    res_by_w, values = [], []
    for w in np.linspace(2.0, 6.0, 60):
        params = {**task_toy.planted_params, "w": float(w)}
        res_by_w.append(check(cfg_toy.model(), params, task_toy.targets, cfg_toy.checker).n_stable)
        values.append(env.evaluate(params, "expert"))
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
    assert env.evaluate(ghost, "expert") > 0.9  # the expert objective is not fooled
    assert env.evaluate(real, "expert") < 0.02
