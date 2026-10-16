# Wilson-Cowan (wc1972) held-out test

Protocol: [NOTES.md](NOTES.md), "wc1972 test protocol" and its AMENDMENT 1, both committed
before the runs (protocol `2834c14`, amendment `d44e816`); the budget came out of the mechanical
rule on the selection set only (NOTES.md, "budget selection after amendment 1": **200**). Objective:
`expert_v2` (condition C). Nothing was changed after seeing these results. Data:
`results/wc_test/` (225 runs: `test.csv`, `test.md`, `bootstrap.md`); run directories in
`runs/wc_test/` (git-ignored).

**Setup.** Tasks `tasks/wc1972_K2_seed1` to `seed9` (9 tasks), seeds 0-4 per task: 45 runs per
condition. Each task carries its own external input P (told to the agent by `get_task()`), the
high target has E in [0.05, 0.45] and the low target is free (almost always the quiescent state,
E ~ 0; see the model card). tol = 0.01. Budget **200** evaluations per run, including the 3
reserved for the submission checks (reserve 3, check the top 3 distinct candidates, submit the
best-scored one that passes). "Success" = the submitted parameters pass the final check.
Conditions: A (random search, ranked by expert_v2), B-cma / B-tpe (naive objective), C-cma / C-tpe
(expert_v2).

**Pre-registered rule for claiming a difference:** the intervals must not overlap in BOTH the
run-level analysis (Wilson 95%) and the task-level analysis (bootstrap over tasks, a task's
seeds kept together).

## Results

| condition | runs | success rate | Wilson 95% | by 25% | by 50% | by 100% | median evals to first success (n) | mean final error | wall-clock per run, mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| A (random, expert_v2) | 45 | 5/45 = 0.11 | [0.05, 0.23] | 0.04 | 0.07 | 0.11 | 78 (5) | 0.158 | 0.4 s (0.8 s) |
| B-cma (naive) | 45 | 12/45 = 0.27 | [0.16, 0.41] | 0.00 | 0.04 | 0.27 | 154 (12) | 0.030 | 36.0 s (54.7 s) |
| B-tpe (naive) | 45 | 14/45 = 0.31 | [0.20, 0.46] | 0.02 | 0.18 | 0.31 | 95 (14) | 0.107 | 44.1 s (82.2 s) |
| C-cma (expert_v2) | 45 | 28/45 = 0.62 | [0.48, 0.75] | 0.09 | 0.18 | 0.62 | 123 (28) | 0.073 | 0.6 s (1.0 s) |
| C-tpe (expert_v2) | 45 | 23/45 = 0.51 | [0.37, 0.65] | 0.13 | 0.38 | 0.53 | 65.5 (24) | 0.132 | 1.1 s (1.6 s) |

"By 25/50/100%" = fraction of runs with a checker-confirmed first success within that share of the
200 evaluations (the 3 submission checks are inside the 200 but are not optimizer evaluations).
For C-tpe 24 runs evaluated a successful point but 23 submitted one.

Task-level analysis (`results/wc_test/bootstrap.md`; resampling the 9 tasks with replacement, seeds
of a task kept together, 20000 resamples):

| condition | success rate (pooled) | tasks solved (>= 3/5 seeds) | bootstrap-over-tasks 95% interval |
|---|---|---|---|
| A | 0.111 | 1/9 | [0.00, 0.24] |
| B-cma | 0.267 | 2/9 | [0.09, 0.49] |
| B-tpe | 0.311 | 1/9 | [0.16, 0.47] |
| C-cma | 0.622 | 6/9 | [0.44, 0.80] |
| C-tpe | 0.511 | 5/9 | [0.31, 0.71] |

**Differences under the pre-registered rule.**

| pair | Wilson intervals apart | bootstrap intervals apart | claimed |
|---|---|---|---|
| A vs C-cma | yes | yes | **yes** |
| A vs C-tpe | yes | yes | **yes** |
| B-cma vs C-cma | yes | no | no |
| B-tpe vs C-cma | yes | no | no |
| A vs B-cma, A vs B-tpe | no | no | no |
| B-cma vs B-tpe, B-cma vs C-tpe, B-tpe vs C-tpe, C-cma vs C-tpe | no | no | no |

So the only claimed differences are that both expert_v2 conditions (C-cma, C-tpe) solve more than
random search. C-cma against the naive conditions separates at run level only, and C-cma against
C-tpe does not separate at either level.

## Per task (successes out of 5 seeds)

| task | A | B-cma | B-tpe | C-cma | C-tpe |
|---|---|---|---|---|---|
| seed1 | 3 | 5 | 4 | 5 | 3 |
| seed2 | 0 | 0 | 1 | 2 | 1 |
| seed3 | 0 | 3 | 2 | 4 | 5 |
| seed4 | 1 | 1 | 0 | 4 | 1 |
| seed5 | 0 | 1 | 2 | 1 | 4 |
| seed6 | 0 | 1 | 2 | 4 | 4 |
| seed7 | 0 | 1 | 1 | 1 | 1 |
| seed8 | 1 | 0 | 0 | 4 | 1 |
| seed9 | 0 | 0 | 2 | 3 | 3 |
| total | 5 | 12 | 14 | 28 | 23 |

## Observations (descriptive)

- Unlike ei_2pop, no condition saturates: the best, C-cma, solves 28 of 45 runs (6 of 9 tasks).
  Its rate on the selection set at this budget was 9/15 = 0.60; here 0.62.
- Tasks seed2 and seed7 are hard for every condition (at most 2 of 5 seeds for any condition);
  seed1 is the easiest (A solves 3 of 5).
- The two naive conditions again cost far more wall-clock (36 to 44 s per run against 0.4 to 1.1 s)
  because the naive objective simulates trajectories.
- C-tpe has the earliest first successes (median 65.5 evaluations, 38% of runs by 50% of the
  budget) but a lower final rate than C-cma; the two are not separated by the rule.
- Caveats: 5 seeds per task and 9 tasks give wide intervals (bootstrap intervals of 0.2 to 0.3
  width); the budget was fixed by the selection rule on 5 other tasks with 3 seeds per cell; in
  these tasks the low target is almost free, so success mostly means reaching a stable active
  state at the high target; a post-hoc analysis of how much choosing a configuration per task could
  add is in `results/selection_headroom.md` (it does not use this test).
