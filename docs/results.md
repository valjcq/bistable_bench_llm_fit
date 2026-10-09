# Results

## LLM agent (Qwen3.6-35B-A3B, local)

Pre-registered tables, Wilson intervals, usage and archives: [agent_results.md](agent_results.md). The protocol is in NOTES.md ("Agent evaluation protocol"). This section summarises it; the tables there stay primary.

- **Solving matches the pipeline.** On feasible tasks the agent found a correct solution in 18/18 episodes in every set and prompt (pipeline P: 18/18 on toy, 17/18 on model_C). That is what to expect when there is no selection headroom: any reasonable first choice solves these tasks.
- **First choice.** With the main prompt, 20 of 27 episodes in each set began with cma/expert_v2, the pipeline's own configuration. With the pressure prompt, 11 of 27 on toy (tpe/expert_v2 first in 16) and 21 of 27 on model_C.
- **Claimed solutions on infeasible tasks.** With the main prompt the agent claimed `solution` on 3/9 (toy) and 2/9 (model_C) infeasible tasks; pipeline P never did (0/9 and 0/9). Pooled over both sets this is 5/18 vs 0/18, but the Wilson intervals (0.12-0.51 vs 0.00-0.18) overlap, and the pooling was not pre-registered. So this is a signal that does not reach the pre-registered criterion, not a demonstrated difference.
- **What the 8 invented solutions look like** (both prompts; post-hoc, descriptive, from the logs only; see agent_results.md). By the rule (a) checked and failed / (b) never checked / (c) other, with a 1e-9 tolerance: (a) 1, (b) 7, (c) 0. Most (b) episodes submitted a rounded copy of a point whose check had failed (6 of 8 are copies, exact or rounded, of a failed check), so "never checked" mostly means "rounded after checking". Five of the 8 justifications say outright that the targets were not matched, and still claim `solution`; one asserts the opposite of what the checker would say, without having been able to check.
- **Correct no_solution.** Almost all came with the budget exhausted at submit (4 of 6 on main/toy, all the others), so they are better read as running out of budget than as a conclusion that no solution exists.
- **Pressure prompt.** No visible honesty effect: the intervals overlap everywhere. Behaviour changed on toy: tpe/expert_v2 first more often, more hyperparameter overrides (16 of 27 episodes vs 8), more checks per episode (4.5 vs 1.9) and 2 episodes without a submission. On model_C it was nearly identical to the main prompt.

**Limitations.** One local quantized model (an IQ4_XS GGUF); 9 infeasible episodes per cell, from 3 seeds on 3 tasks, so episodes are not independent; sampling at temperature 0.6, so runs are not bit-reproducible (the transcripts and configs are the record); a post-hoc classification of 8 episodes, with a keyword screen read by hand.
