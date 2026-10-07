# Phase 0 validation (no LLM)

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

**The project plan's toy validation checklist is not in this repository, so the list below is
PROVISIONAL: items I derived from what Phase 0 should establish. Replace it with the real
checklist; the evidence column can then be re-pointed.**

| # | Item | Done | Evidence |
|---|---|---|---|
| 1 | Planted parameters solve every task; targets are in-window and separated | yes | `tests/test_tasks.py` |
| 2 | Checker agrees with simulation on the toy | yes | `tests/test_checker.py` (cross-check) |
| 3 | Budget counts every call, refuses at the limit, cannot be bypassed | yes | `tests/test_budget_env.py`; every grid run used exactly 200 evaluations (A, B, C), the dummy agent 195 or 101 (section 2) |
| 4 | Simulation horizon is capped, refused for free | yes | `test_simulate_beyond_t_max_is_refused_for_free` |
| 5 | Each optimizer solves a toy task within budget | yes | `tests/test_optimizers.py`; B-cma 18/25, B-tpe 12/25, C-cma 11/25, C-tpe 2/25 (section 2) |
| 6 | Random search is a weak baseline on the toy at budget 200 | yes | A: 0/25, Wilson [0.00, 0.13] (section 2) |
| 7 | Optimizers never see the model, checker or planted parameters | yes | `test_optimizers_never_import_the_environment_side`, `test_optimizers_never_see_planted_params`, `test_get_task_never_reveals_the_planted_parameters`, `test_run_writes_a_complete_log` (no planted values in run dirs) |
| 8 | Same seed gives identical results and logs (minus timestamp, latency) | yes | `test_rerun_is_identical_except_volatile_fields`, `test_dummy_agent_rerun_gives_identical_logs`; the 75 A/B-cma/C-cma runs of the budget-200 sweep reproduce the main grid exactly (success, first success, final error) |
| 9 | A run can be reconstructed from events.jsonl + config.json | yes | `test_run_is_reconstructible_from_events_and_config` |
| 10 | Tool layer: invalid calls are free, flagged, return errors; nothing works after submit | yes | `tests/test_tools.py` |
| 11 | Tool schemas are valid JSON Schema and the runtime validator agrees with `jsonschema` | yes | `test_schemas_pass_the_official_metaschema_and_agree_with_the_runtime_validator` |
| 12 | Scripted agent completes an episode with a complete, grouped log (toy and ei_2pop) | yes | `test_dummy_agent_completes_an_episode_with_a_complete_log`; 25 toy runs and 1 ei_2pop run in section 2 |
| 13 | ei_2pop runs end to end for every condition | yes (smoke only) | section 3 |
| 14 | Expert objective is at least as useful as the naive one | **no** | section 2: B beats C |

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
  The flat-in-monostable-regions weakness is documented in NOTES.md and fits this, but I have
  not tested that explanation.
- **The scripted agent (cma then tpe, expert) is no better than C-cma**, with a lower rate
  (5/25). It spends half its budget per optimizer and uses the weak TPE-expert pair for the
  second half.
- **Task difficulty is uneven.** Task seed3 is almost never solved (B: 1 of 5 per optimizer,
  C: 0), seed0 is solved only by B-cma.
- Condition A almost never succeeds at 200, which is the intended property for the toy.

## 3. ei_2pop smoke (budget 1000, task seed0, seed 0, one run per condition)

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

## 4. Toy budget sweep (5 tasks x 5 seeds per cell)

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

## 5. Reproduce

```
python -m bench.batch configs/exp_toy_1pop.yaml --tasks "tasks/toy_1pop_*.json" \
  --conditions A B:cma B:tpe C:cma C:tpe S --seeds 0-4 --jobs 20 --csv results/phase0/toy_grid.csv
python -m bench.batch configs/exp_ei_2pop.yaml --tasks tasks/ei_2pop_K2_seed0.json \
  --conditions A B:cma B:tpe C:cma C:tpe S --seeds 0 --csv results/phase0/ei_smoke.csv
python -m bench.batch configs/exp_toy_1pop.yaml --tasks "tasks/toy_1pop_*.json" \
  --conditions A B:cma C:cma --seeds 0-4 --budget 100 --csv results/phase0/toy_sweep_100.csv
```
(and 200, 500, 1000). Bit-identical results need the locked environment (`uv.lock`).
