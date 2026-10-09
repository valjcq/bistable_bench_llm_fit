# Design decisions

The decisions that shaped the project, with the evidence in the repository. The dated, line-by-line log is [../NOTES.md](../NOTES.md); formal records are in [decisions/](decisions/). The narrative is in [STORY.md](STORY.md).

| Decision | Why | Evidence in the repo |
|---|---|---|
| Keep the environment separate from any agent | The agent must not be able to influence the referee | Budgeted interface only (`src/bench/env.py`, `tools.py`); leak tests (`tests/test_leaks.py`, `tests/test_transcript_leaks.py`); [environment.md](environment.md) section 9 |
| Structured weights, Dale's law, partial observation | Matches published models; free weights would make placing states trivial | [environment.md](environment.md) section 4; [model_cards/ei_2pop.md](model_cards/ei_2pop.md) |
| Exclude saturated targets when planting (5%-95% window) | Near either end the sigmoid is flat, so a 0.02 tolerance there is meaningless | [environment.md](environment.md) section 5; `configs/exp_*.yaml` |
| Write the protocol before every test | Results must not shape the rules | Protocol sections in NOTES.md state "written and committed BEFORE"; changes appear as labelled AMENDMENT sections; [protocol.md](protocol.md) |
| Replace expert v1 with v2 | v1 was flat on plateaus and did worse than the naive objective on the toy | [decisions/0001-expert-objective.md](decisions/0001-expert-objective.md); [phase0.md](phase0.md); [results.md](results.md) section 1 |
| One submission rule for all baselines (reserve 3 evaluations, check the top 3 distinct candidates) | A baseline that never checks its answer is a weak comparison | NOTES.md "Submission rule"; the rule changed 2 of 175 toy runs ([phase0.md](phase0.md)) |
| Wilson-Cowan from a model card | Code adapted to the paper's equations, never the paper's numbers adapted to our code | [model_cards/wilson_cowan_1972.md](model_cards/wilson_cowan_1972.md); `tests/test_wc1972.py` (`paper` and `computed` markers) |
| Measure selection headroom before building the agent | An LLM is only warranted if no fixed setup wins everywhere | [../results/selection_headroom.md](../results/selection_headroom.md); [results.md](results.md) section 4 |
| Test honesty with provably impossible tasks | The risk of an LLM is a false solution; a proof makes "impossible" certain | [honesty.md](honesty.md); `tests/test_infeasible.py` |
| Neutral labels and renamed parameters | Stop the agent from recalling the paper | [agent.md](agent.md); leak tests |
| Log every agent decision in full | Results must be checkable later | JSONL logs, `scripts/show_episode.py`, [../archives/MANIFEST.md](../archives/MANIFEST.md) |

## Decision records

- [0001: The expert objective: remove v1, keep v2](decisions/0001-expert-objective.md). Git tag `phase0-expert-v1` marks the last commit that still contains `expert_v1` (see [../results/phase0/README.md](../results/phase0/README.md)). That commit is `c87d08f`, which also contains the ei_2pop test protocol and the ei_2pop budget-selection runs; it is not a snapshot of Phase 0 alone.
