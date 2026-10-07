# ei_2pop held-out test

Protocol: [NOTES.md](../NOTES.md), "ei_2pop test protocol" and its AMENDMENT, both committed
before this run (amendment in commit `209750f`). Objective: `expert_v2` (condition C), chosen
and fixed on the toy ([decisions/0001-expert-objective.md](decisions/0001-expert-objective.md));
nothing was tuned or changed after seeing these results. Data: `results/ei_test/` (225 runs,
`test.csv`, `test.md`); run directories in `runs/ei_test/` (git-ignored).

**Setup.** Tasks `ei_2pop_K2_seed1` to `seed9` (9 tasks; seed0, used in the Phase 0 smoke run,
is excluded), seeds 0-4 per task, so 45 runs per condition. Budget **500** evaluations per run,
which includes the 3 reserved for the submission checks (reserve 3, check the top 3 distinct
candidates, submit the best-scored one that passes). Budget 500 was Val's choice and an explicit
exception to the pre-registered budget rule (on the selection set random search succeeded in
2/15 runs at 500, against a threshold of at most 1/15). "Success" = the submitted parameters pass
the final check. A difference is only called a difference when the Wilson 95% intervals do not
overlap.

## Results

| condition | runs | success rate | Wilson 95% | by 25% | by 50% | by 100% | median evals to first success (n) | mean final error | wall-clock per run, mean (median) |
|---|---|---|---|---|---|---|---|---|---|
| A (random, expert_v2) | 45 | 8/45 = 0.18 | [0.09, 0.31] | 0.00 | 0.04 | 0.18 | 348.5 (8) | 0.237 | 0.4 s (0.4 s) |
| B-cma (naive) | 45 | 36/45 = 0.80 | [0.66, 0.89] | 0.09 | 0.33 | 0.80 | 279.5 (36) | 0.065 | 105.4 s (104.1 s) |
| B-tpe (naive) | 45 | 31/45 = 0.69 | [0.54, 0.80] | 0.20 | 0.49 | 0.69 | 184 (31) | 0.090 | 108.1 s (105.9 s) |
| C-cma (expert_v2) | 45 | 44/45 = 0.98 | [0.88, 1.00] | 0.18 | 0.67 | 1.00 | 217 (45) | 0.001 | 0.8 s (0.8 s) |
| C-tpe (expert_v2) | 45 | 35/45 = 0.78 | [0.64, 0.87] | 0.33 | 0.60 | 0.80 | 161 (36) | 0.066 | 2.6 s (2.6 s) |

"By 25/50/100%" is the fraction of runs with a checker-confirmed first success within that share
of the 500 evaluations (the 3 submission checks are counted in the 500 but not as optimizer
evaluations). For C-tpe, 36 runs evaluated a successful point at some time but 35 submitted
one.

**Differences by the pre-registered rule (non-overlapping Wilson intervals).**
- A is below every other condition (A [0.09, 0.31] against B-tpe [0.54, 0.80], B-cma
  [0.66, 0.89], C-tpe [0.64, 0.87], C-cma [0.88, 1.00]).
- C-cma is above B-tpe ([0.88, 1.00] against [0.54, 0.80]) and above C-tpe ([0.64, 0.87]).
- Not called a difference (intervals overlap): C-cma against B-cma ([0.88, 1.00] against
  [0.66, 0.89]); C-tpe against B-tpe; B-cma against B-tpe; B-cma against C-tpe.

## Per task (successes out of 5 seeds)

| task | A | B-cma | B-tpe | C-cma | C-tpe |
|---|---|---|---|---|---|
| seed1 | 1 | 5 | 5 | 5 | 5 |
| seed2 | 1 | 1 | 0 | 5 | 1 |
| seed3 | 4 | 5 | 5 | 5 | 5 |
| seed4 | 2 | 5 | 5 | 5 | 5 |
| seed5 | 0 | 5 | 2 | 5 | 4 |
| seed6 | 0 | 3 | 3 | 5 | 4 |
| seed7 | 0 | 5 | 5 | 5 | 4 |
| seed8 | 0 | 3 | 3 | 5 | 5 |
| seed9 | 0 | 4 | 3 | 4 | 2 |
| total | 8 | 36 | 31 | 44 | 35 |

## Observations (descriptive; no conclusions beyond the rule above)

- C-cma succeeded in 44 of 45 runs; its one failure is on task seed9. It is the only condition
  that solves task seed2 (5/5), where the others solve 0 to 1 of 5.
- Random search solves 8 runs, concentrated on tasks seed1 to seed4 (seed3: 4 of 5, seed4: 2 of
  5); on tasks seed5 to seed9 it solves none.
- The two naive conditions cost about 105 s per run against 0.4 to 2.6 s for the others: the
  naive objective simulates each target's trajectory, the v2 objective does not.
- On the toy (development set) C-cma with v2 reached 24/25 and B-cma 18/25; on this held-out
  set the order is the same (44/45 against 36/45) but the intervals overlap, so by the rule it
  is not a difference here. For TPE, v2 against naive was 18/25 against 12/25 on the toy and
  is 35/45 against 31/45 here.
- Caveats: the 5 seeds of a task are not independent samples of the task distribution, so the
  intervals are optimistic for generalisation across tasks; the budget (500) is an exception to
  the pre-registered budget rule; random search is at 0.18 here against 0.13 on the selection
  set.
