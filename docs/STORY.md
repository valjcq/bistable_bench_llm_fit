# Bistable Fit Bench: the design journey

As of 2026-10-09, by Valentin Jacquin. Written from my own narrative notes and checked against the repository; where the two differed, the repository won (see [the list at the end](#differences-between-this-narrative-and-the-repository)). Numbers link to their source in [results.md](results.md).
 
We set out to test whether an LLM agent helps fit multistable neural models. The work showed that a well-designed objective, not the agent, does the solving; the agent study became a test of whether it picks the right optimizer and whether it invents solutions.
 
## At a glance
 
Each step answered a question raised by the one before.
 
1. **Frame the task.** The agent solves an inverse problem: given target rates, find parameters that make them stable states.
2. **Build a deterministic referee.** Checker, simulator, scores and budget, separate from any agent, tested against each other.
3. **Run baselines on a toy.** The first expert score lost to plain simulation; diagnosing why gave expert v2, which won.
4. **Test on held-out tasks.** ei_2pop: v2 with CMA-ES solved 44/45 runs. Too easy to leave room for an agent.
5. **Add a published model.** Wilson–Cowan 1972, reproduced from the paper. Hard only at a tight budget.
6. **Ask if an LLM is warranted.** Selection headroom was zero: one fixed setup was best everywhere.
7. **Pivot the agent study.** Measure optimizer choice and honesty, using tasks that provably have no solution.
8. **Run a local LLM.** Qwen solved every solvable task and picked the best setup first in most episodes, but claimed solutions on some impossible tasks.
## Starting question
 
The agent's job is the inverse problem, not the math. Knowing the parameters, the fixed points follow from the equations: that forward problem is the checker's job. The task gives target rates and asks for parameters that make them stable fixed points.
 
- **The agent is a researcher at an optimization toolbox.** It picks and configures optimizers, reads results, notices bad fits and decides when to submit. It does not guess numbers itself.
- **The toy is solvable by hand.** In one population, the fixed-point condition is linear in the parameters, so the toy only validates the harness.
- **Multiple populations.** A state became a vector of rates, and targets can cover only some populations (partial observation), as in experiments that record mainly excitatory cells.
- **Structured weights and Dale's law.** A few named parameters fill the connectivity matrix, as in published models; connections out of an inhibitory population are negative.
- **Placing states is easy; stability is the hard part.** Target states are linear in the weights and inputs, so the difficulty comes from stability, unobserved populations and bounds.
## Building an environment we can trust
 
The environment is the referee: deterministic, separate from any agent, and checked against itself. Claude Code built it in four reviewed checkpoints.
 
- **Checker.** In 1D, an exact grid scan with root refinement finds every fixed point. In 2D and up, Newton from each target is exact and cheap; a global multi-start search finds extra states but is reported as non-exhaustive. Stability comes from Jacobian eigenvalues, which depend on the time constants.
- **Simulator.** Fixed-step RK4, so runs are bit-reproducible; the settle criterion scales with the time constants.
- **Cross-check.** Simulations from many starting states must end only on states the checker found. This is the test a reviewer trusts.
- **Three scores kept apart.** A naive score (simulate and compare), an expert score (based on the checker), and the final metric used only for reporting.
- **Budget.** Optimizers see only budgeted calls; one parameter set evaluated = one evaluation.
- **Planted tasks.** Hidden parameters are drawn until the system is multistable; their stable states become targets. A window excludes saturated targets, where any strong weights would match.
- **Reproducibility.** Seeds, config hashes, frozen task files, a lockfile and git tags.
Reviews caught real issues: the toy's threshold and input were redundant; a grid point landing exactly on a root; a matching rule that could miss a valid assignment; and a crash for parameter sets with no stable state, found by a limit-cycle test.
 
## Baselines and the objective-design finding
 
The first expert score lost to plain simulation; fixing its two traps made it the best setup. Conditions: A = random search, B = optimizer + naive score, C = optimizer + expert score. Baseline hyperparameters are fixed; the agent may tune them.
 
Expert v1 had two traps, confirmed by classifying failed runs:
 
- **Saddle on a target.** An unstable fixed point near a target lowered the score without solving anything.
- **Cliff at the tolerance.** A target just outside tolerance cost about 1, so near misses gave no signal.
Expert v2 was written down and committed before any run. It scores the fixed-point residual at each target plus a penalty if the target is unstable. It is smooth everywhere and needs no fixed-point search.
 
| Toy, budget 200 (25 runs each) | CMA-ES | TPE |
| --- | --- | --- |
| Naive score (B) | 17 | 12 |
| Expert v1 (C) | 11 | 2 |
| Expert v2 (C2) | 24 | 18 |
| Random search (A) | 0 | — |

Source: `results/phase0/toy_grid_sr_summary.csv`, with the final submission rule. Before that rule the same grid gave B-cma 18 and expert v2 with TPE 17 (`toy_grid_summary.csv`, `toy_grid_v2_summary.csv`); the other cells are identical.
 
These are development numbers: v2 was designed after seeing these tasks. v1 was then removed from the code, kept at the tag `phase0-expert-v1`, and its replacement documented in a decision record. All baselines also adopted one submission rule: check the top 3 candidates before submitting.
 
## Harder models and held-out tests
 
Expert v2 held up on tasks it was never designed on, and the published model was hard only at a tight budget. Each test followed the same discipline: choose the budget on separate selection tasks, write the protocol, run the test set once.
 
**ei_2pop (E–I pair, 6 free parameters).** The first budget rule ("random search rarely succeeds") could not be met at any budget and was replaced openly. At budget 500, C-cma solved 44/45 runs: too easy to leave room for an agent.
 
**Wilson–Cowan 1972 (8 free parameters).** A model card was written from the paper. The code was generalized (refractory term, shifted sigmoid, state bounds per model) and reproduces the paper's qualitative results: bistability (Fig. 4), three stable states (Figs. 7–8), a limit cycle (Fig. 11). Three design choices followed:
 
- **Box.** Narrowed to avoid sigmoids far from the paper's regime.
- **Window.** The low state is always quiescent (E ≈ 0), so only the high target is constrained.
- **Tolerance.** 2% of each model's range (0.01 for Wilson–Cowan).
C-cma solved every selection run from budget 500 up, so the grid was extended down and the rule picked 200.
 
| Held-out test | A | B-cma | B-tpe | C-cma | C-tpe |
| --- | --- | --- | --- | --- | --- |
| ei_2pop, budget 500 (45 runs) | 8 | 36 | 31 | 44 | 35 |
| Wilson–Cowan, budget 200 (45 runs) | 5 | 12 | 14 | 28 | 23 |
 
On Wilson–Cowan, the difficulty comes from the budget, not the landscape: with enough evaluations, C-cma always succeeds.
 
## Is an LLM warranted?
 
No, not on these tasks: one fixed setup was the best choice everywhere, so an agent could at most tie it.
 
**Why every model came out easy.** The tasks ask few numbers of many parameters: 2 target rates for 6 free parameters in ei_2pop, effectively 1 for 8 in Wilson–Cowan. That leaves large families of solutions. A new paper model, or more populations, posed the same way would likely be just as easy. Difficulty comes from task structure: several states or conditions at once, narrow disconnected solution regions, or no solution at all.
 
**The legitimacy test.** The question is not whether an optimizer alone can solve the task. It is whether one fixed setup works on every task. The algorithm-selection literature measures this as the gap between:
 
- the **single best solver** (the best fixed setup on average), and
- the **virtual best solver** (an oracle picking the best setup per task).
That gap is the most any selector, LLM or not, can gain.
 
| Model | Single best | Gap (plain) | Gap (conservative) |
| --- | --- | --- | --- |
| Toy, budget 200 | C-cma | 0.000 | 0.000 |
| ei_2pop, budget 500 | C-cma | 0.000 | 0.000 |
| Wilson–Cowan, budget 200 | C-cma | 0.067 | 0.000 |
 
This analysis was post hoc, over 5 configurations and 3–5 seeds per cell. Its conclusion still stands: on these tasks, an LLM agent adds no solving power over a fixed expert pipeline.
 
## The pivot: honesty and optimizer choice
 
The agent study became two questions: does it pick the best setup, and does it invent solutions? Budget stopped being the difficulty knob.
 
**Tasks with provably no solution, mixed in with normal ones (3 of 9 per set):**
 
- **Toy.** Two stable states can never sit on the same side of 0.5, for any parameters. Derived analytically and checked numerically; such tasks look like any other.
- **Wilson–Cowan.** With the paper's refractory term, every fixed point has E < 1/2. A high target at 0.52–0.60 is impossible.
**Setup:**
 
- **Two answers.** Submit "solution" with parameters, or "no_solution". Explanations were not required.
- **No hints.** Tool descriptions say what each tool computes, never which works best. Models are labelled model_A/B/C, and Wilson–Cowan's parameters are renamed, so the agent cannot recall the paper. Leak tests cover all of it.
- **Reference.** A fixed pipeline (CMA-ES with expert v2, "no_solution" if the check fails), which never invents by construction.
- **Full logs.** Every turn, reasoning text, tool call and result is logged and can be rebuilt as a readable episode; the runs are archived with checksums.
**Local Qwen results** (Qwen3.6-35B-A3B, 4-bit, on llama.cpp; 3 seeds per task; a pressure variant adds one sentence about an upcoming experiment):
 
| Set | Prompt | Solvable: solved | Impossible: claimed "solution" | Pipeline: claimed "solution" |
| --- | --- | --- | --- | --- |
| Toy | main | 18/18 | 3/9 | 0/9 |
| Toy | pressure | 18/18 | 2/9 | 0/9 |
| Wilson–Cowan | main | 18/18 | 2/9 | 0/9 |
| Wilson–Cowan | pressure | 18/18 | 1/9 | 0/9 |
 
- **Solving matches the pipeline,** as the zero headroom predicted.
- **It reads the tools.** With the main prompt, it started with CMA-ES on expert v2 in 20 of 27 episodes per set.
- **It sometimes claims solutions to impossible tasks.** 5 of 18 with the main prompt, against 0 for the pipeline. This is a signal: with 9 episodes per cell it does not reach the pre-registered criterion.
- **Pressure changed behavior, not honesty.** More TPE, overrides and checks on the toy, and 2 episodes that never submitted.
- **Nuance.** At least one claimed "solution" came with a justification admitting the targets were not matched. Most correct refusals came after the budget ran out: "could not find one", not "none exists".
## Key decisions and why
 
| Decision | Why | Evidence |
| --- | --- | --- |
| Keep the environment separate from any agent | The agent must not be able to influence the referee | Budgeted interface only; leak tests |
| Structured weights, Dale's law, partial observation | Matches published models; free weights make placing states trivial | Linear fixed-point condition |
| Exclude saturated targets when planting | Saturated states match almost any strong weights | 5%–95% planting window ([environment.md](environment.md) section 5) |
| Write the protocol before every test | Results must not shape the rules | Commits dated before runs; rule changes labelled |
| Replace expert v1 with v2 | v1 rewarded saddles near targets and had a cliff at the tolerance | Classified v1 failures ([phase0.md](phase0.md)); toy 11/25 → 24/25 with CMA-ES |
| Same submission rule for all baselines | A baseline that never checks its answer is a weak comparison | Top-3 check; 2 of 175 runs changed |
| Wilson–Cowan from a model card | Code is adapted to the paper's equations, never the paper's numbers to our code | Paper's qualitative results reproduced |
| Measure selection headroom before building the agent | An LLM is only warranted if no fixed setup wins everywhere | Gap 0 on toy and ei_2pop |
| Test honesty with provably impossible tasks | The risk of an LLM is a false solution; a proof makes "impossible" certain | Toy theorem; E < 1/2 bound |
| Neutral labels and parameter names | Stop the agent from recalling the paper | Leak tests on every agent-visible output |
| Log every agent decision in full | Results must be checkable later | Episode viewer; archives with checksums |
 
## Open points
 
The current conclusion: on these tasks, an LLM agent adds no solving power over a fixed expert pipeline, and it brings a non-zero risk of reporting false solutions.
 
- [ ] Read all 8 claimed solutions on impossible tasks myself. A post-hoc classification exists (a: checked and failed, b: never checked, c: other; tally 1/7/0, plus hand notes on the justifications), labelled descriptive and small-sample in [agent_study.md](agent_study.md); no conclusion is drawn from it.
- [ ] Decide whether to run one closed API model on the same tasks and seeds (main prompt, 54 episodes, about 2–3 million input tokens).
- [x] The agent section is in the results documents ([results.md](results.md), [agent_results.md](agent_results.md)) and the README.
- [ ] Possible later extensions, not planned: Wilson–Cowan hysteresis tasks, a 3-population tristable model with partial observation.
Limitations: one local quantized model, 9 impossible episodes per cell, sampling at temperature 0.6, toy and Wilson–Cowan only. Full list in [agent_study.md](agent_study.md).

## Differences between this narrative and the repository

Where the notes and the repo differed, I used the repo.

1. **Toy table.** The notes had B-cma 18 and expert v2 + TPE 17 (before the submission rule). The table above uses the final rule (17 and 18); the pre-rule figures are in the footnote.
2. **"95% of toy high states were above 0.96 before the window".** I found no source for this in the repo, so it is removed.
3. **Open points 1 and 3** were out of date: the post-hoc classification and the agent section in the results documents already existed.
4. **Tag `phase0-expert-v1`.** The notes say v1 "was kept at the tag". The tag is the last commit that still contains `expert_v1` (`c87d08f`); it also contains the ei_2pop test protocol and its budget-selection runs, so it is not a snapshot of Phase 0 alone.
5. **Pipeline count.** "8 of 36 vs 0": the pipeline was run once per set, so its reference is 0 of 18 episodes shown against both prompts.
6. **Limitations.** The notes have no numbered section 9; the limitations are the last line of the notes, expanded in [agent_study.md](agent_study.md).
