# Results

Every number below is copied by script from a result file in the repository; the last column (or the line under each table) names that file. Nothing here comes from the narrative alone.

Conditions (from [environment.md](environment.md) and [phase0.md](phase0.md)): **A** random search; **B** optimizer on the naive (simulation) score; **C** optimizer on the expert score. `-cma` is CMA-ES, `-tpe` is TPE (Optuna). In the Phase 0 files the expert score is `expert_v1` as `C` and `expert_v2` as `C2`; in every later file (ei_2pop, Wilson-Cowan tests) `C` is `expert_v2`, and `expert_v1` no longer exists in the code. A run succeeds if the submitted parameters pass the final check. Intervals are Wilson 95%.

## Protocols

The protocols stay in [../NOTES.md](../NOTES.md), where they were committed before the runs they govern; this table points to them rather than copying them. Each held-out test followed the same discipline: choose the budget on separate **selection tasks**, write the protocol, then run the **test set** once. Rule changes are labelled amendments, written and committed before the next runs.

| Stage | Selection tasks | Test tasks | Budget | Protocol (NOTES.md section) | Report |
|---|---|---|---|---|---|
| Toy, expert v2 (development) | none (development set: `tasks/toy_1pop_K2_seed0..4`) | none | 200 | "expert_v2: pre-registered definition", "Submission rule" | [phase0.md](phase0.md) |
| ei_2pop | `tasks/selection/ei_2pop_K2_seed100..104` | `tasks/ei_2pop_K2_seed1..9` | 500 (exception to the budget rule, see the amendment) | "ei_2pop test protocol", "AMENDMENT", "test: done" | [ei2pop_test.md](ei2pop_test.md) |
| Wilson-Cowan 1972 | `tasks/selection/wc1972_K2_seed100..104` | `tasks/wc1972_K2_seed1..9` | 200 (selected by the rule after amendment 1) | "wc1972 test protocol", "AMENDMENT 1", "test: done" | [wc_test.md](wc_test.md) |
| Agent evaluation | none | `tasks/honesty/` (toy seeds 300-308, Wilson-Cowan seeds 400-408) | toy 500, model_C 1000 | "Agent evaluation protocol" | [agent_results.md](agent_results.md), [STORY.md](STORY.md#the-pivot-honesty-and-optimizer-choice) |

Statistics used throughout: Wilson 95% intervals on counts; a difference is claimed only when the intervals do not overlap. Section 4 below (headroom) was not pre-registered.

## 1. Toy (`toy_1pop`), budget 200: expert v1 vs v2 (development tasks)

These are development numbers: expert v2 was designed after seeing these tasks. 5 tasks x 5 seeds = 25 runs per condition, all conditions use the final submission rule (check the top 3 distinct candidates before submitting).

| condition | runs | successes | success rate | Wilson 95% | source file |
|---|---|---|---|---|---|
| A | 25 | 0 | 0.00 | [0.00, 0.13] | `results/phase0/toy_grid_sr_summary.csv` |
| B-cma | 25 | 17 | 0.68 | [0.48, 0.83] | `results/phase0/toy_grid_sr_summary.csv` |
| B-tpe | 25 | 12 | 0.48 | [0.30, 0.67] | `results/phase0/toy_grid_sr_summary.csv` |
| C-cma (expert v1) | 25 | 11 | 0.44 | [0.27, 0.63] | `results/phase0/toy_grid_sr_summary.csv` |
| C-tpe (expert v1) | 25 | 2 | 0.08 | [0.02, 0.25] | `results/phase0/toy_grid_sr_summary.csv` |
| C2-cma (expert v2) | 25 | 24 | 0.96 | [0.80, 0.99] | `results/phase0/toy_grid_sr_summary.csv` |
| C2-tpe (expert v2) | 25 | 18 | 0.72 | [0.52, 0.86] | `results/phase0/toy_grid_sr_summary.csv` |

Footnote, before the submission rule (same tasks, same budget): A 0/25; B-cma 18/25; B-tpe 12/25; C-cma 11/25; C-tpe 2/25; S 5/25 (`results/phase0/toy_grid_summary.csv`, expert v1 and naive) and C2-cma 24/25; C2-tpe 17/25 (`results/phase0/toy_grid_v2_summary.csv`, expert v2). Condition S is a scripted dummy agent. Per the repo, the submission rule changed 2 of 175 runs (`docs/phase0.md`). The reproduction of v1 numbers needs git tag `phase0-expert-v1` (see [phase0 README](../results/phase0/README.md)).

## 2. Held-out tests

Budget chosen on separate selection tasks (seeds 100-104), the protocol written and committed before the test run, the test set run once. 9 test tasks x 5 seeds = 45 runs per condition. Protocols: [above](#protocols).

### ei_2pop, budget 500

| condition | runs | successes | success rate | Wilson 95% | source file |
|---|---|---|---|---|---|
| A | 45 | 8 | 0.18 | [0.09, 0.31] | `results/ei_test/test_summary.csv` |
| B-cma | 45 | 36 | 0.80 | [0.66, 0.89] | `results/ei_test/test_summary.csv` |
| B-tpe | 45 | 31 | 0.69 | [0.54, 0.80] | `results/ei_test/test_summary.csv` |
| C-cma | 45 | 44 | 0.98 | [0.88, 1.00] | `results/ei_test/test_summary.csv` |
| C-tpe | 45 | 35 | 0.78 | [0.64, 0.87] | `results/ei_test/test_summary.csv` |

Details and a bootstrap over tasks: [ei2pop_test.md](ei2pop_test.md), `results/ei_test/bootstrap.md`.

### Wilson-Cowan 1972, budget 200

| condition | runs | successes | success rate | Wilson 95% | source file |
|---|---|---|---|---|---|
| A | 45 | 5 | 0.11 | [0.05, 0.23] | `results/wc_test/test_summary.csv` |
| B-cma | 45 | 12 | 0.27 | [0.16, 0.41] | `results/wc_test/test_summary.csv` |
| B-tpe | 45 | 14 | 0.31 | [0.20, 0.46] | `results/wc_test/test_summary.csv` |
| C-cma | 45 | 28 | 0.62 | [0.48, 0.75] | `results/wc_test/test_summary.csv` |
| C-tpe | 45 | 23 | 0.51 | [0.37, 0.65] | `results/wc_test/test_summary.csv` |

Details: [wc_test.md](wc_test.md), `results/wc_test/bootstrap.md`.

## 3. Budget selection (selection tasks, 15 runs per cell)

The selection rule needed a baseline between 0.30 and 0.80 and random search at most 1 of 15 (ei_2pop: no budget met it, the rule was amended before the test run; Wilson-Cowan: the grid was extended downward, amendment 1). Sources: `results/ei_selection/sel_*_summary.csv`, `results/wc_selection/sel_*_summary.csv`, NOTES.md.

### ei_2pop

| budget | A | B-cma | C2-cma | source file |
|---|---|---|---|---|
| 500 | 2/15 | 11/15 | 14/15 | `results/ei_selection/sel_500_summary.csv` |
| 1000 | 2/15 | 10/15 | 14/15 | `results/ei_selection/sel_1000_summary.csv` |
| 2000 | 4/15 | 10/15 | 14/15 | `results/ei_selection/sel_2000_summary.csv` |

### Wilson-Cowan 1972

| budget | A | B-cma | C-cma | source file |
|---|---|---|---|---|
| 100 | 3/15 | 2/15 | 3/15 | `results/wc_selection/sel_100_summary.csv` |
| 200 | 3/15 | 5/15 | 9/15 | `results/wc_selection/sel_200_summary.csv` |
| 300 | 5/15 | 9/15 | 10/15 | `results/wc_selection/sel_300_summary.csv` |
| 500 | 5/15 | 11/15 | 15/15 | `results/wc_selection/sel_500_summary.csv` |
| 1000 | 6/15 | 12/15 | 15/15 | `results/wc_selection/sel_1000_summary.csv` |
| 2000 | 11/15 | 12/15 | 15/15 | `results/wc_selection/sel_2000_summary.csv` |

## 4. Algorithm-selection headroom (post-hoc)

Post-hoc analysis of existing results; nothing was pre-registered. Gap = mean over tasks of (best configuration per task) minus (best single configuration), both estimated from the same 3 to 5 seeds, so the per-task best is optimistically biased; the conservative gap credits a different configuration only when it wins by more than one seed. Source: `results/selection_headroom.md` (script `scripts/selection_headroom.py`).

| model | tasks | SBS | mean SBS | mean VBS | gap (plain) | gap (conservative) |
|---|---|---|---|---|---|---|
| toy_1pop (budget 200) | 5 | C-cma | 0.960 | 0.960 | 0.000 | 0.000 |
| ei_2pop (budget 500) | 9 | C-cma | 0.978 | 0.978 | 0.000 | 0.000 |
| wc1972 selection set (budget 200, 3 seeds) | 5 | C-cma | 0.600 | 0.667 | 0.067 | 0.000 |
| all tasks pooled (each model with its own SBS) | 19 | - | 0.874 | 0.891 | 0.018 | 0.000 |

Different models have different configuration sets (WC has no TPE) and different seed counts, so the pooled row is a task-weighted summary, not a like-for-like estimate.

## 5. LLM agent (Qwen3.6-35B-A3B, local) vs the fixed pipeline P

Counts of episodes per outcome; the intervals, usage, optimizer-choice tables and excerpts are in [agent_results.md](agent_results.md). Impossible tasks: 3 tasks x 3 seeds = 9 per cell. Solvable tasks: 6 x 3 = 18 per cell. Sets: `toy` = toy_1pop (budget 500), `model_C` = Wilson-Cowan 1972 under a neutral label (budget 1000). Harness errors are excluded (0 in all cells).

| set | prompt | agent | solvable: correct solution | impossible: claimed `solution` | impossible: correct `no_solution` | impossible: no submission | source file |
|---|---|---|---|---|---|---|---|
| toy | main | Qwen3.6-35B-A3B | 18/18 | 3/9 | 6/9 | 0/9 | `results/agent_eval/main_toy.csv` |
| toy | main | pipeline P | 18/18 | 0/9 | 9/9 | 0/9 | `results/agent_eval/reference_toy.csv` |
| toy | pressure | Qwen3.6-35B-A3B | 18/18 | 2/9 | 5/9 | 2/9 | `results/agent_eval/pressure_toy.csv` |
| toy | pressure | pipeline P | 18/18 | 0/9 | 9/9 | 0/9 | `results/agent_eval/reference_toy.csv` |
| model_C | main | Qwen3.6-35B-A3B | 18/18 | 2/9 | 7/9 | 0/9 | `results/agent_eval/main_model_C.csv` |
| model_C | main | pipeline P | 17/18 | 0/9 | 9/9 | 0/9 | `results/agent_eval/reference_model_C.csv` |
| model_C | pressure | Qwen3.6-35B-A3B | 18/18 | 1/9 | 8/9 | 0/9 | `results/agent_eval/pressure_model_C.csv` |
| model_C | pressure | pipeline P | 17/18 | 0/9 | 9/9 | 0/9 | `results/agent_eval/reference_model_C.csv` |

Pooled over both sets and both prompts: Qwen claimed `solution` on 8 of 36 impossible-task episodes (3+2 with the main prompt, 2+1 with the pressure prompt). Pipeline P was run once per set and is shown against both prompts: it claimed `solution` on 0 of its 18 impossible-task episodes (computed from the rows above). The pooling was not pre-registered; with 9 episodes per cell and Wilson intervals that overlap in each pre-registered comparison, this is a signal, not a demonstrated difference. See [STORY.md](STORY.md#the-pivot-honesty-and-optimizer-choice).
