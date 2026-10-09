# Protocols, budgets, selection and test sets

This page is an index. The protocols themselves stay in [../NOTES.md](../NOTES.md), where they were committed before the runs they govern; a copy here could drift from them.

Discipline used for each held-out test: choose the budget on separate **selection tasks**, write the protocol, then run the **test set** once. Rule changes are labelled amendments, written and committed before the next runs.

| Stage | Selection tasks | Test tasks | Budget | Protocol (NOTES.md section) | Report |
|---|---|---|---|---|---|
| Toy, expert v2 (development) | none (development set: `tasks/toy_1pop_K2_seed0..4`) | none | 200 | "expert_v2: pre-registered definition", "Submission rule" | [phase0.md](phase0.md) |
| ei_2pop | `tasks/selection/ei_2pop_K2_seed100..104` | `tasks/ei_2pop_K2_seed1..9` | 500 (exception to the budget rule, see the amendment) | "ei_2pop test protocol", "AMENDMENT", "test: done" | [ei2pop_test.md](ei2pop_test.md) |
| Wilson-Cowan 1972 | `tasks/selection/wc1972_K2_seed100..104` | `tasks/wc1972_K2_seed1..9` | 200 (selected by the rule after amendment 1) | "wc1972 test protocol", "AMENDMENT 1", "test: done" | [wc_test.md](wc_test.md) |
| Agent evaluation | none | `tasks/honesty/` (toy seeds 300-308, Wilson-Cowan seeds 400-408) | toy 500, model_C 1000 | "Agent evaluation protocol" | [agent_results.md](agent_results.md), [agent_study.md](agent_study.md) |

Selection and budget numbers: [results.md](results.md) sections 2 and 3. The post-hoc headroom analysis, which was not pre-registered, is section 4 there.

Statistics used throughout: Wilson 95% intervals on counts; a difference is claimed only when the intervals do not overlap. Success means the submitted parameters pass the final check.
