# Phase 0 validation (no LLM)

> **Produced at git tag `phase0-expert-v1`.** All Phase 0 results below used the first expert
> objective (`expert_v1`; "expert" and condition C mean v1, C2 means v2), which was later removed:
> see [decisions/0001-expert-objective.md](decisions/0001-expert-objective.md). Test names and
> commands cited in this page refer to the code at that tag (`git checkout phase0-expert-v1`);
> the results files are in `results/phase0/` (see its README).

Everything here comes from runs committed under [results/phase0/](../results/phase0/) (CSV per
run, summary per table, produced by `python -m bench.batch`; commands at the end). Run
directories with full logs are in `runs/phase0/` (git-ignored). Settings: toy budget 200
unless stated, baseline defaults of NOTES.md ("Baseline settings"), 5 toy tasks
(`toy_1pop_K2_seed0..4`) x seeds 0-4 = 25 runs per condition.

Column meanings (full definitions in the docstring of `src/bench/batch.py`): *success* = the
submitted parameters pass the final check; *by 25/50/100%* = fraction of runs whose first
checker-confirmed success happened within that share of the budget; *median* is over runs
that had a success (n shown); *final error* = mean over targets of the submitted error,
unmatched target = 1.0.

## 1. Checklist

The Phase 0 toy validation checklist from the project plan, each item with its evidence.

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | The checker labels known bistable and known monostable parameter sets correctly (unit tests) | ticked | `tests/test_checker.py::test_bistable_hand_checked`, `::test_monostable_cannot_realize_two_targets`, `::test_checker_and_simulator_agree_on_toy`; `tests/test_model.py::test_toy_bistable_hand_checked`, `::test_toy_small_w_is_monostable` |
| 2 | The score is consistent: better fits score better | ticked, with a caveat | `tests/test_scoring.py::test_expert_separates_success_from_failure_and_counts_unmatched` (all matched <= tol, each unmatched >= penalty/K), `::test_expert_is_zero_at_the_planted_solution`, `::test_expert_is_never_constant_across_a_monostable_bistable_sweep`. Caveat: this holds for the expert objective's ordering, but the expert objective is flat on plateaus (section 3) and condition C does worse than B (section 2); the naive objective can be fooled (`::test_naive_objective_can_be_fooled_by_a_slow_transient_past_a_fold`). "Consistent" is tested, "useful for search" is not established. |
| 3 | Random search, Bayesian optimization and CMA-ES each solve the toy within the budget in most seeds | **not met as written**; **amended criterion met** (amendment accepted by Val after seeing the results) | numbers below |
| 4 | A scripted "dummy agent" (fixed sequence of tool calls) runs the full loop and produces logs | ticked | `tests/test_tools.py::test_dummy_agent_completes_an_episode_with_a_complete_log` (toy and ei_2pop), `::test_dummy_agent_rerun_gives_identical_logs`; 25 toy runs and 1 ei_2pop run with complete logs (sections 2 and 4) |
| 5 | Results are identical when rerun with the same seed | ticked | `tests/test_runner.py::test_rerun_is_identical_except_volatile_fields`, `tests/test_tools.py::test_dummy_agent_rerun_gives_identical_logs`, `tests/test_optimizers.py::test_same_seed_gives_identical_history`, `tests/test_determinism.py`; the 75 A/B-cma/C-cma runs of the budget-200 sweep reproduce the main grid exactly (success, first success, final error) |

**Item 3, numbers (budget 200, 25 runs per row = 5 tasks x 5 seeds; "most seeds" read as more
than half).** Bayesian optimization is represented by TPE (Optuna), not a Gaussian-process BO.

| optimizer | with expert objective | with naive objective |
|---|---|---|
| random search | 0/25 (A), Wilson [0.00, 0.13] | not run |
| TPE | 2/25 (C-tpe), [0.02, 0.25] | 12/25 (B-tpe) = 0.48, [0.30, 0.67] |
| CMA-ES | 11/25 (C-cma) = 0.44, [0.27, 0.63] | 18/25 (B-cma) = 0.72, [0.52, 0.86] |

Only B-cma exceeds half. Random search fails at every budget up to 1000 (3/25 at 1000), TPE
reaches 0.48 at best, and CMA-ES with the expert objective plateaus at 15/25 at budget 1000
(section 5).

*Why.* The tasks are the in-window ones (targets between 0.05 and 0.95 of the rate range,
separated by more than 3 tol), which make the toy harder than planned. With the saturated
hand-checked targets [0, 1] used in the first optimizer smoke test, all three optimizers solved
the task for at least one of 5 seeds, but I did not measure rates there, so I cannot say how
much easier it was. Random search is expected to fail on a three-parameter problem with a
2% tolerance.

*Amended criterion.* **This is a criterion change made after seeing the results; it was
accepted by Val.** New wording: "At least one optimizer solves the toy in most seeds." On the
numbers: B-cma solves 18/25 = 72% at budget 200 (met), and random search solves 0/25 as
expected. The original wording stays visible above as "not met as written".

## 2. Toy results (budget 200, 5 tasks x 5 seeds)

| condition | runs | success rate | Wilson 95% | by 25% | by 50% | by 100% | median evals to first success (n) | mean final error |
|---|---|---|---|---|---|---|---|---|
| A (random, expert) | 25 | 0/25 = 0.00 | [0.00, 0.13] | 0.00 | 0.00 | 0.00 | n/a (n=0) | 0.312 |
| B-cma (naive) | 25 | 18/25 = 0.72 | [0.52, 0.86] | 0.00 | 0.08 | 0.72 | 147 (n=18) | 0.013 |
| B-tpe (naive) | 25 | 12/25 = 0.48 | [0.30, 0.67] | 0.04 | 0.16 | 0.48 | 115.5 (n=12) | 0.033 |
| C-cma (expert) | 25 | 11/25 = 0.44 | [0.27, 0.63] | 0.00 | 0.12 | 0.44 | 120 (n=11) | 0.153 |
| C-tpe (expert) | 25 | 2/25 = 0.08 | [0.02, 0.25] | 0.00 | 0.04 | 0.08 | 114.5 (n=2) | 0.192 |
| S (scripted agent) | 25 | 5/25 = 0.20 | [0.09, 0.39] | 0.00 | 0.12 | 0.20 | 97 (n=5) | 0.177 |

Source: `results/phase0/toy_grid.md`. Evaluations used: 200 for every A/B/C run; the scripted
agent used 195 in 22 runs (two optimizer runs, 5 left unspent by design) and 101 in 3 runs
(its first check succeeded, so it submitted without the second run).

Successes per task (of 5 seeds), from `toy_grid.csv`:

| task | A | B-cma | B-tpe | C-cma | C-tpe | S |
|---|---|---|---|---|---|---|
| seed0 | 0 | 3 | 0 | 0 | 0 | 0 |
| seed1 | 0 | 5 | 4 | 4 | 0 | 3 |
| seed2 | 0 | 5 | 4 | 5 | 0 | 2 |
| seed3 | 0 | 1 | 1 | 0 | 0 | 0 |
| seed4 | 0 | 4 | 3 | 2 | 2 | 0 |

**Surprises**
- **B beats C**, for both optimizers: B-cma 18/25 vs C-cma 11/25 (intervals overlap
  [0.52, 0.86] vs [0.27, 0.63]); B-tpe 12/25 vs C-tpe 2/25 (intervals [0.30, 0.67] and
  [0.02, 0.25] do not overlap). The expert objective was expected to be the better signal.
  Section 3 reports what the logs show; it does not settle why.
- **The scripted agent (cma then tpe, expert) is no better than C-cma**, with a lower rate
  (5/25). It spends half its budget per optimizer and uses the weak TPE-expert pair for the
  second half.
- **Task difficulty is uneven.** Task seed3 is almost never solved (B: 1 of 5 per optimizer,
  C: 0), seed0 is solved only by B-cma.
- Condition A almost never succeeds at 200, which is the intended property for the toy.

## 3. Why C does worse than B: observations from the existing logs

Source: `scripts/phase0_diagnosis.py` (no new optimizer runs; the checker calls below are
off-budget analysis of points the runs already evaluated), output in
`results/phase0/diagnosis.md`. These are observations, not conclusions. Samples are small
(25 runs per condition, 5 tasks).

### (a) Final best expert score of FAILED C runs

| condition | failed runs | < tol | [tol, 0.5) | [0.5, 0.6) | [0.6, 0.75) | [0.75, 1.0) | >= 1.0 |
|---|---|---|---|---|---|---|---|
| C-cma | 14 | 0 | 0 | 3 | 3 | 2 | 6 |
| C-tpe | 23 | 0 | 0 | 4 | 1 | 3 | 15 |

C-cma: median 0.809, min 0.517, max 1.141; targets matched at the best point (checker): {0: 6, 1: 8} (key = number of targets matched).

C-tpe: median 1.027, min 0.513, max 1.141; targets matched at the best point (checker): {0: 15, 1: 8} (key = number of targets matched).

### Plateau level (1 + separation) / 2

- C-cma: 5 of 14 failed runs end within 0.01 of their task's (1 + separation) / 2.
- C-tpe: 4 of 23 failed runs end within 0.01 of their task's (1 + separation) / 2.

### (b) What the evaluated points are (checker, off-budget), 25 runs x 200 evals

| condition | monostable (1 stable state) | no stable state | bistable+ (>=2 stable) | success |
|---|---|---|---|---|
| B-cma | 0.498 | 0.000 | 0.502 | 0.040 |
| C-cma | 0.532 | 0.000 | 0.468 | 0.024 |

### (c) Best-so-far expert score of stuck C-cma runs

| run | @1 | @10 | @25 | @50 | @100 | @150 | @200 | last improvement | monostable evals | evals with score >= 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| task0 seed0 | 1.505 | 1.128 | 1.128 | 0.722 | 0.722 | 0.722 | 0.722 | 38 | 172/200 | 200/200 |
| task3 seed0 | 1.481 | 1.118 | 0.809 | 0.809 | 0.809 | 0.809 | 0.809 | 25 | 57/200 | 200/200 |


**What I see**
- Every failed C run ends with a best score of at least 0.5, as the objective guarantees (an
  unmatched target costs at least penalty/K = 0.5). They do not cluster just above 0.5: C-cma
  has 3 of 14 in [0.5, 0.6) and 6 of 14 at 1.0 or more, with median 0.809; C-tpe has 4 of 23
  in [0.5, 0.6) and 15 of 23 at 1.0 or more. At the best point the checker finds 8 of 14
  C-cma and 8 of 23 C-tpe runs with one target matched, and 6 and 15 with none matched.
- Some failures sit exactly on the plateau where a monostable state matches one target: 5 of
  14 C-cma and 4 of 23 C-tpe failed runs end within 0.01 of (1 + separation) / 2 of their
  task. That is not most of the failures.
- C-cma did not spend a much larger share of evaluations on monostable points than B-cma
  (53.2% vs 49.8%), and bistable points were similar (46.8% vs 50.2%). Of all evaluated
  points, 2.4% (C-cma) and 4.0% (B-cma) passed the checker.
- In the two stuck C-cma runs shown, the best score stopped improving at evaluation 38 and 25
  of 200 and every one of the 200 evaluations scored 0.5 or more. Their final scores equal the
  plateau level of their task, (1 + 0.445) / 2 = 0.722 and (1 + 0.619) / 2 = 0.809.
- What the logs do not show: why the B runs are not stuck in the same way, since B-cma's
  evaluated points are similarly often monostable. I have not examined the naive objective's
  landscape here.

### Classification of failed C runs (final best params)

Priority order a > b > c > d; each failed run counted once, non-exclusive counts in brackets
(source: `results/phase0/diagnosis.md`).

| condition | failed runs | (a) monostable plateau | (b) near miss, stable state within 3 tol of an unmatched target | (c) nearest fixed point to an unmatched target is unstable | (d) other |
|---|---|---|---|---|---|
| C-cma | 14 | 5 (5) | 4 (4) | 5 (6) | 0 (0) |
| C-tpe | 23 | 4 (4) | 7 (7) | 10 (15) | 2 (2) |

**Reading the classification as evidence for the two v1 traps.** The expert_v1 objective has
two structural traps, and the classes (b) and (c) are the failures they predict:
- *Saddle on a target* (class c): when a target coincides with an unstable fixed point, v1
  counts the target as unmatched and charges the distance to the nearest fixed point, with no
  signal about how to make that point stable. In class (c) the fixed point nearest to an
  unmatched target is unstable: 5 of 14 C-cma runs (6 counting overlaps) and 10 of 23 C-tpe
  runs (15 counting overlaps).
- *Cliff at tol* (class b): a stable state within 3 tol of an unmatched target is close, yet
  v1 charges it the full unmatched cost (at least 1 against at most tol when matched), so
  there is no gradient toward the last bit of accuracy. 4 of 14 C-cma and 7 of 23 C-tpe runs.
- *Plateau* (class a, the flat monostable plateau of the earlier analysis): 5 of 14 and 4 of
  23.
Together (b) and (c) account for 9 of 14 failed C-cma runs and 17 of 23 failed C-tpe runs
(exclusive counts), so the two traps are the majority of failures, and the plateau a minority.
Only 0 and 2 runs fall in "other". This is descriptive evidence about where v1 runs end; it
does not show that removing the traps is what makes a different objective work. expert_v2
(section 6) was designed to remove both traps: its score at each target is continuous in the
parameters (no tolerance, no unmatched cost) and an unstable target is penalized in
proportion to how unstable it is. Its results are development numbers (section 6).

## 4. Per-task difficulty


| task | low target | high target | low-edge dist | high-edge dist | separation | planted margin_eig | planted margin_dist | A | B-cma | B-tpe | C-cma | C-tpe | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | 0.273 | 0.718 | 0.223 | 0.232 | 0.445 | 0.0129 | 0.207 | 0 | 3 | 0 | 0 | 0 | 0 |
| seed1 | 0.177 | 0.949 | 0.127 | 0.001 | 0.771 | 0.0158 | 0.100 | 0 | 5 | 4 | 4 | 0 | 3 |
| seed2 | 0.101 | 0.920 | 0.051 | 0.030 | 0.819 | 0.0487 | 0.358 | 0 | 5 | 4 | 5 | 0 | 2 |
| seed3 | 0.210 | 0.829 | 0.160 | 0.121 | 0.619 | 0.0222 | 0.238 | 0 | 1 | 1 | 0 | 0 | 0 |
| seed4 | 0.052 | 0.851 | 0.002 | 0.099 | 0.799 | 0.0260 | 0.164 | 0 | 4 | 3 | 2 | 2 | 0 |

edge dist = distance of the target to the window edge (0.05 / 0.95); separation = high - low (the guard is 3 tol = 0.06); margins are those of the planted solution (`margin_eig` = slowest decay rate, `margin_dist` = distance to the unstable point).

Observations: the two hardest tasks (seed0 and seed3) are the ones with targets farthest from
the window edges at the low end (0.223 and 0.160); seed0 has the smallest separation (0.445)
and the smallest planted eigenvalue margin (0.0129). The easiest, seed2, has the largest
planted margins (0.0487, 0.358). Two tasks have a target almost on a window edge (seed1: 0.001
from the upper edge; seed4: 0.002 from the lower edge) and are solved by several conditions,
so closeness to the edge does not by itself make a task hard. With 5 tasks these are
descriptions, not tested relationships.

## 5. Toy budget sweep (5 tasks x 5 seeds per cell; expert_v2 in section 6)

Success of the submitted parameters, with Wilson 95% intervals.

| budget | A | B-cma | C-cma |
|---|---|---|---|
| 100 | 0/25 [0.00, 0.13] | 2/25 [0.02, 0.25] | 3/25 [0.04, 0.30] |
| 200 | 0/25 [0.00, 0.13] | 18/25 [0.52, 0.86] | 11/25 [0.27, 0.63] |
| 500 | 1/25 [0.01, 0.20] | 25/25 [0.87, 1.00] | 14/25 [0.37, 0.73] |
| 1000 | 3/25 [0.04, 0.30] | 25/25 [0.87, 1.00] | 15/25 [0.41, 0.77] |

Median evaluations to first success (n runs with a success): A 464 (1) at 500 and 662 (3) at
1000; B-cma 158 (25) at both 500 and 1000; C-cma 131.5 (14) at 500 and 132 (15) at 1000.
Fractions by 25/50/100% of the budget are in `results/phase0/toy_sweep_*.md`.

Reading, limited to what the table shows: B-cma solves all 25 from budget 500; C-cma
plateaus at 14-15 of 25 between 500 and 1000, so more budget does not help it. At 1000 the
unsolved C-cma runs are concentrated on two tasks (seed3: 0 of 5, seed0: 1 of 5, per-task
counts from `toy_sweep_1000.csv`). The default toy budget (200) is unchanged; the choice is
yours.

## 6. expert_v2 (candidate replacement of the expert objective)

The definition and the evaluation plan were written into NOTES.md and committed (`784d4a0`)
before any v2 run. The toy is the development set; expert_v2 was **not** run on ei_2pop at
scale (ei_2pop is the test set), and delta (0.01) and lambda (1.0) were not tuned. The
objective was designed after seeing the Phase 0 results on these same toy tasks, so the toy
numbers below are development results, not a clean test.

Toy grid, budget 200, 5 tasks x 5 seeds (`results/phase0/toy_grid_v2.md`), next to the v1 and
B results of section 2:

| condition | success rate | Wilson 95% | by 25% | by 50% | by 100% | median evals to first success (n) | mean final error |
|---|---|---|---|---|---|---|---|
| B-cma (naive) | 18/25 = 0.72 | [0.52, 0.86] | 0.00 | 0.08 | 0.72 | 147 (18) | 0.013 |
| B-tpe (naive) | 12/25 = 0.48 | [0.30, 0.67] | 0.04 | 0.16 | 0.48 | 115.5 (12) | 0.033 |
| C-cma (expert_v1) | 11/25 = 0.44 | [0.27, 0.63] | 0.00 | 0.12 | 0.44 | 120 (11) | 0.153 |
| C-tpe (expert_v1) | 2/25 = 0.08 | [0.02, 0.25] | 0.00 | 0.04 | 0.08 | 114.5 (2) | 0.192 |
| **C2-cma (expert_v2)** | 24/25 = 0.96 | [0.80, 0.99] | 0.00 | 0.04 | 0.96 | 126.5 (24) | 0.024 |
| **C2-tpe (expert_v2)** | 17/25 = 0.68 | [0.48, 0.83] | 0.00 | 0.20 | 0.72 | 126.5 (18) | 0.072 |

Successes per task (of 5 seeds):

| task | B-cma | C-cma (v1) | C2-cma (v2) | B-tpe | C-tpe (v1) | C2-tpe (v2) |
|---|---|---|---|---|---|---|
| seed0 | 3 | 0 | 4 | 0 | 0 | 1 |
| seed1 | 5 | 4 | 5 | 4 | 0 | 3 |
| seed2 | 5 | 5 | 5 | 4 | 0 | 5 |
| seed3 | 1 | 0 | 5 | 1 | 0 | 3 |
| seed4 | 4 | 2 | 5 | 3 | 2 | 5 |

Budget sweep for CMA-ES (`results/phase0/toy_sweep_v2_*.md`; A, B-cma and C-cma are from
section 5):

| budget | A | B-cma | C-cma (v1) | C2-cma (v2) |
|---|---|---|---|---|
| 100 | 0/25 | 2/25 | 3/25 | 1/25 [0.01, 0.20] |
| 200 | 0/25 | 18/25 | 11/25 | 24/25 [0.80, 0.99] |
| 500 | 1/25 | 25/25 | 14/25 | 25/25 [0.87, 1.00] |
| 1000 | 3/25 | 25/25 | 15/25 | 25/25 [0.87, 1.00] |

**Observations**
- With CMA-ES, expert_v2 solves 24/25 at budget 200 against 11/25 for v1 and 18/25 for the
  naive objective; the v2 interval [0.80, 0.99] lies above the v1 interval [0.27, 0.63] and
  overlaps B's [0.52, 0.86]. At 500 and 1000 it solves all 25. At budget 100
  it is the worst of the three CMA-ES conditions with a success (1/25), so it needs more than
  100 evaluations here.
- The one CMA-ES failure at 200 is task seed0, seed 4 (final error 0.504). In the per-task
  table seed0 and seed3, the tasks v1 never solved, are solved 4/5 and 5/5.
- With TPE, v2 solves 17/25 against 2/25 for v1 and 12/25 for B-tpe. Its first-success rate
  by 100% of the budget (0.72) is above its success rate (0.68): in one run (task seed1,
  seed 4) a successful point was evaluated at evaluation 78 but the submitted point, the best
  by v2 score, was not a success (final error 0.013, i.e. a target just out of tolerance). The
  best v2 score does not always identify a checker success.
- Time to first success is similar across objectives (medians 115 to 147 evaluations).

Not shown here: any ei_2pop evidence for v2, any effect of delta or lambda, and whether v2's
advantage depends on the toy having no unobserved population (the unobserved-population
branch is only unit-tested via the ei_2pop zero-score check).

## 7. Submission rule (reserve 3 evaluations, check the top 3)

Decided before any test run and applied to every baseline (A, B, C, C2): 3 evaluations are
reserved from the run's own budget, so the optimizer gets 197 of the 200; at the end the 3
best DISTINCT parameter sets by the condition's objective are checked with the checker, and
the best-scored one that passes is submitted, else the best-scored one
(`baseline.submission_checks`, `choose_submission` in `src/bench/runner.py`). Rerun of the toy
grid (budget 200, 5 tasks x 5 seeds, `results/phase0/toy_grid_sr.md`), next to the old rule
(submit the best-scored point, all 200 evaluations to the optimizer; sections 2 and 6):

| condition | old rule | new rule | Wilson 95% (new) |
|---|---|---|---|
| A | 0/25 | 0/25 | [0.00, 0.13] |
| B-cma | 18/25 | 17/25 | [0.48, 0.83] |
| B-tpe | 12/25 | 12/25 | [0.30, 0.67] |
| C-cma (v1) | 11/25 | 11/25 | [0.27, 0.63] |
| C-tpe (v1) | 2/25 | 2/25 | [0.02, 0.25] |
| C2-cma (v2) | 24/25 | 24/25 | [0.80, 0.99] |
| C2-tpe (v2) | 17/25 | 18/25 | [0.52, 0.86] |

Per-run comparison: only two of 175 runs changed. B-cma on task seed0, seed 1 went from success
to failure: its only success was found at evaluation 200, the last one, which the reserve
takes away. C2-tpe on task seed1, seed 4 went from failure to success: it is the run of
section 6 whose best v2 score was not a success; the top-3 check found the successful point.
Everything else is identical in outcome. v2 results remain development numbers.

## 8. ei_2pop smoke (budget 1000, task seed0, seed 0, one run per condition)

| condition | success | evals used | first success (eval) | final error |
|---|---|---|---|---|
| A | no | 1000 | - | 0.017 |
| B-cma | yes | 1000 | 62 | 0.000 |
| B-tpe | yes | 1000 | 953 | 0.014 |
| C-cma | no | 1000 | - | 0.509 |
| C-tpe | no | 1000 | - | 0.094 |
| S | no | 995 | - | 0.095 |

One run each says nothing about rates; it only shows that every condition runs on ei_2pop.
The wall clock of the naive conditions is about 90 to 120 s per run versus about 20 s for the
expert ones (`ei_smoke.csv`). The target was that A rarely succeeds on ei_2pop; one run
cannot confirm that.

## 9. Reproduce

```
python -m bench.batch configs/exp_toy_1pop.yaml --tasks "tasks/toy_1pop_*.json" \
  --conditions A B:cma B:tpe C:cma C:tpe S --seeds 0-4 --jobs 20 --csv results/phase0/toy_grid.csv
python -m bench.batch configs/exp_ei_2pop.yaml --tasks tasks/ei_2pop_K2_seed0.json \
  --conditions A B:cma B:tpe C:cma C:tpe S --seeds 0 --csv results/phase0/ei_smoke.csv
python -m bench.batch configs/exp_toy_1pop.yaml --tasks "tasks/toy_1pop_*.json" \
  --conditions A B:cma C:cma --seeds 0-4 --budget 100 --csv results/phase0/toy_sweep_100.csv
```
(and 200, 500, 1000). Section 6: the same commands with `--conditions C2:cma C2:tpe` (grid) and
`--conditions C2:cma --budget N` (sweep). Sections 3 and 4: `python scripts/phase0_diagnosis.py` (needs `runs/phase0/grid`).
Bit-identical results need the locked environment (`uv.lock`).
