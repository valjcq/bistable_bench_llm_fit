# Notes: decision log

Internal log of design decisions and caveats (the explanation of the environment is in
[docs/environment.md](docs/environment.md)).

## Status
All four checkpoints are implemented. Things to keep in mind when building optimizers/agent:
- `tasks/*.json` contain the HIDDEN planted params next to the targets: never hand these files
  to an agent; only `Environment.from_task` (which reads targets + seed) should see them.
- `Environment.final_report` is the one unbudgeted method (success metric for the harness). It
  must not be exposed to optimizers.
- The expert objective is graded but nearly flat once the system is monostable (the only signal
  is the surviving fixed point drifting), with a jump of penalty/K whenever a target becomes
  unmatched. The naive objective is smoother there but can be fooled (test_scoring.py).
- Planting cost: ~260 random draws/task for ei_2pop, ~900 for toy_1pop (cap 5000, heavy tail).

## Facts worth remembering
- `tau_I` is a plain value under `fixed:` in `configs/ei_2pop.yaml` (default 5). A slow-inhibition
  variant is a config change (see the limit-cycle regression test), not a new instance.
- tau_I = 5 makes oscillations very rare (~0.1% of random parameter sets vs ~2% at tau_I = 20).
  It is a difficulty knob for later: slow inhibition creates oscillatory regions where the naive
  (simulation-based) objective never settles and the fixed-point checker finds no stable state.
- `uv.lock` pins the environment. Bit-identity of results (simulator, checker, tasks) is only
  guaranteed within ONE locked environment (same numpy/scipy/libm/BLAS builds); across machines or
  library versions, expect last-digit differences. Reinstall with `uv sync` to reproduce.

## Decisions and caveats (from the checkpoint 4 review)
- **Naive objective can be fooled (evidence for condition B's known weakness).** Test
  `test_naive_objective_can_be_fooled_by_a_slow_transient_past_a_fold` (tests/test_scoring.py):
  toy with w=10, k=1 just past the fold (I = I* + 1e-3) is monostable, so the checker fails it,
  but a simulation started low is still sitting on the vanished low state (a "ghost") when time T
  ends. The naive objective reports a near-perfect fit (better than a genuinely successful
  nearby solution) while the expert objective is not fooled. (Condition B = the naive-objective
  condition.)
- **Expert objective is weak in monostable regions.** It is graded (no ties) but the only signal
  there is the surviving fixed point drifting, plus a jump of penalty/K when a target becomes
  unmatched. No change for now. Revisit if CMA-ES with expert_objective underperforms
  naive_objective.
- **Target ordering convention.** `make_task` sorts targets by increasing mean observed rate, and
  the naive objective pairs target i with initial condition `naive_ic_fractions[i]` (low IC <->
  lowest target). Unsorted targets would make the pairing arbitrary.
- **`final_report`** is the only unbudgeted method and is not part of the optimizer-facing
  interface (`env.BUDGETED_METHODS`); later only `submit()` will call it.
- **Toy planting cap** is 20000 (`max_attempts`); the toy needs ~900 draws on average with a heavy
  tail. `max_attempts` is excluded from `generation_hash` (it only decides when to give up).

## Optimizers (checkpoint 5)
- `src/bench/optimizers/` never imports model/checker/env (a test checks the imports). The
  environment side is `src/bench/optimizer_runs.py` (`RunRegistry`: unit -> params, seeds, runs).
- **Converged flag** (same for all optimizers): best-so-far improved by < 1e-6 over the last
  max(20, 20% of run budget) evaluations, and more than that many evaluations were done. It is
  informational: runs spend their whole budget unless `stop_on_converged=True` (off for baselines
  so conditions stay budget-matched; then `stopped_reason="converged"`).
- CMA-ES: Optuna samples trial 0 independently, so we enqueue x0 (or the cube centre) as trial 0;
  CMA then starts a cloud centred on x0. `x0`/`sigma0` are deprecated in Optuna >= 4.9 (removed
  in v6); we pin optuna 5.0.0 and silence the warning.
- Simulation time cap: `Environment.simulate` refuses T > T_max (default 200 * max tau), for free.
  The low-level `simulate_batch` is uncapped (tests use long horizons).

## Experimental conditions
Budget-matched: every condition gets the same evaluation budget.

| | Who searches | Objective |
|---|---|---|
| A | random search | expert |
| B | optimizer (CMA-ES or TPE) | naive |
| C | same optimizer | expert (the human baseline) |
| D | LLM agent, open model | chosen by the agent |
| E | same agent, closed API model | chosen by the agent |

At the end of a baseline episode (A-C) the best parameters by the condition's objective are submitted.
- Optuna v6 removes x0/sigma0 from CmaEsSampler; upgrading requires changing cma.py.

## Baseline settings
Default hyperparameters (spec defaults, ranges in brackets). Used unchanged by conditions A-C and
MUST NOT change once experiments start. The agent (D-E) sees them as defaults in list_optimizers()
and may override any of them within the spec ranges, at the cost of budget.
- random: sampling = "uniform" ["uniform", "lhs"]
- tpe: n_startup_trials = 10 [1, 1000]; multivariate = True [True, False]
- cma: sigma0 = 0.25 [0.01, 0.5]; popsize = null = library default 4 + 3 ln d [null or 2..200]

## Possible later option (not built)
An automated-tuning baseline (a fixed sweep over a few settings per optimizer, budget-matched), to
answer "would simple automatic tuning match the agent?".

## Observation from the checkpoint 5 smoke test
Expert objective, budget 200, toy tasks seed0-4, 5 seeds each (successes / 5): random 1,0,0,0,0;
tpe 0,0,0,0,2; cma 1,1,2,0,2. Condition B/C on the toy is hard at 200 evaluations; revisit the
toy budget after the first runs (as planned).

## Logging, runner, batch (checkpoint 6)
- Budgets are now placeholders: toy 200, ei_2pop 1000 (`budget.max_evaluations` in the exp
  configs). Experiment configs have a `baseline: {optimizer: cma|tpe}` section (B and C use it;
  `--optimizer` overrides it per run). It is part of the config hash.
- Log format and the "same run" rule (ignore only timestamp and latency): `src/bench/logging.py`.
  `Environment` events now carry `budget_remaining`.
- Summary definitions (success rate, first success and whether its checks are counted, mean final
  error): docstring of `src/bench/batch.py`. Short version: first-success checks are post-hoc
  analysis and are NOT counted against the budget; expert score <= tol is itself success, naive
  candidates (score <= tol) are verified with the checker.
- `config.json` in a run directory holds the resolved experiment config, targets and seed (never
  planted params); `config_from_dict(..., resolved=True)` rebuilds the config from it.

## Checkpoint 6 review fixes
- First success: score <= tol only selects candidates for BOTH objectives; the checker confirms
  (off-budget). Note: with the real expert objective an out-of-tolerance target costs >= 1.0, so
  an average like (0.00 + 0.03)/2 cannot occur; the regression test covers the hypothetical anyway.
- CMA `popsize` range is [4, 200] (the cmaes library documents no minimum).
- `--optimizer` is applied to the config before anything is written, so config.json and the config
  hash record the optimizer used; it is an error for condition A.
- Summary adds the fraction of runs with a confirmed first success by 25/50/100% of the budget,
  and prints n next to the median.

## Planned for checkpoint 8 (in addition to the spec)
Toy budget sweep at 100, 200, 500, 1000 evaluations for A, B-cma, C-cma (5 tasks x 5 seeds), with
the success table. The toy default budget stays 200 until Val chooses it from that table.

## Tool layer and dummy agent (checkpoint 7)
- `src/bench/tools.py`: `Episode.call(name, arguments)` is the one entry point (validate against the
  tool's JSON Schema, run, convert failures to `{"error": {...}}`, log). `tool_specs()` returns
  name/description/`input_schema` per tool; parameter names and bounds come from the task.
- Error types: invalid_argument (free, logged `invalid: true`), budget_exhausted, episode_over
  (every tool after submit, also logged invalid), unknown_tool. Unexpected exceptions are bugs and
  propagate on purpose.
- Log roles: tool-call lines and `check`/`simulate` calls the agent makes itself are `agent`; the
  evaluations inside run_optimizer are `optimizer`; `final_report` is `harness`. A tool call
  produces its environment lines first, then its own line (so submit follows final_report).
- Schemas are checked with `src/bench/schema.py`, a small validator for the keyword subset we use
  (no `jsonschema` dependency was added). A test keeps the schemas inside that subset.
- Dummy agent (`agent/scripted.py`): second run gets the rest of the budget minus MARGIN=5, so an
  episode where the check fails spends budget - 5 evaluations.

## Review fixes before checkpoint 8
- Config validation: `expert_penalty / K` must exceed `tol` (the "score <= tol means all matched"
  equivalence needs it). First-success still confirms candidates with the checker.
- Logging: `call_id` on the line a tool call writes; `parent_call_id` on every line written while
  it ran (environment/optimizer events). Group a log by call with these.
- Schemas declare draft 2020-12 (`$schema`); `jsonschema` is a DEV dependency only; tests check
  the metaschema and that the runtime validator (kept) agrees with jsonschema on examples.
- **For the agent stage:** an unexpected exception in `Episode.call` is a bug. The runner must
  record it as `harness_error`, exclude that run from statistics and count such runs separately.

## expert_v2: pre-registered definition (written and committed BEFORE any v2 run)
Motivation (from `docs/phase0.md` section 3): the current expert objective (now called
**expert_v1**) is flat on plateaus and does worse than the naive objective in the Phase 0 toy
grid. expert_v2 is the candidate replacement. Definition, fixed before any run:

For each target k:
- complete the target state r*: observed populations = the target values; each unobserved
  population solves its own fixed-point equation given the others (for ei_2pop:
  r_I = F_I(w_IE * E* - w_II * r_I + I_I), unique on [0, r_max], solved with brentq).
- residual_k = max abs over observed populations of (-r* + F(W r* + I)).
- stab_k = max(0, max Re eig(J(r*)) * max(tau) + delta), delta in config (default 0.01).
- score = mean over targets of (residual_k + lam * stab_k), lam in config (default 1.0).

No fixed-point search, no simulation. One call = one evaluation. Config names:
`scoring.v2_delta` (0.01) and `scoring.v2_lambda` (1.0). Success is still decided only by the
checker (first-success confirmation as before). expert_v2 is exposed as an objective option in
run_optimizer; the old objective is renamed `expert_v1` (kept, with its results; logs written
before the rename say "expert", which means v1).

Evaluation plan, fixed in advance: C-cma and C-tpe with expert_v2 on the toy grid (budget 200,
5 tasks x 5 seeds), and C-cma with expert_v2 in the budget sweep (100/200/500/1000), reported
next to the v1 and B results whatever the outcome. The toy is the development set: ei_2pop is
NOT run with v2 at scale (it is the test set); no parameter (delta, lam) is tuned on ei_2pop.
Implementation notes (not changes to the definition): at most one unobserved population is
supported (the only case in the shipped models); the unobserved root is bracketed on
[0, r_max] by the sign of g at the ends.

## expert_v2 results (toy, development set)
Per the pre-registered plan: C2-cma 24/25 at budget 200 (C-cma v1 11/25, B-cma 18/25), C2-tpe
17/25 (v1 2/25); C2-cma sweep 1/25, 24/25, 25/25, 25/25 at 100/200/500/1000. Details and caveats
(toy is the dev set, no ei_2pop run, best-v2-score is not always a checker success) in
`docs/phase0.md` section 6. Objective names: naive, expert_v1 (was "expert"), expert_v2;
condition C2 = C with expert_v2. Event logs from before the rename say "expert" = v1.

## Submission rule (all baselines), decided before any test run
`baseline.submission_checks = 3` (hash-recorded; 0 = the Phase 0 rule). Applies to A, B, C, C2:
3 evaluations are reserved from the run's own budget; at the end the 3 best DISTINCT candidates by
the condition's objective are checked with the checker (budgeted `check` calls, role `harness`,
logged as `select_submission`); the best-scored one that passes is submitted, else the best-scored
one. Phase 0 numbers under the old rule are kept next to the new ones in `docs/phase0.md`.
Conventions to remember: expert_v2 supports at most one unobserved population; logs written before
the rename say "expert", which means expert_v1.

## ei_2pop test protocol (written and committed BEFORE any ei_2pop selection or test run)
Config: `configs/exp_ei_2pop.yaml` unchanged except the budget set per run below; all
hyperparameters at their Baseline settings; `scoring.v2_delta = 0.01`, `v2_lambda = 1.0`; the
submission rule (`baseline.submission_checks = 3`) for every condition. Nothing about objectives,
optimizers or settings is changed between the budget selection and the test run; a bug found
in between is reported, not silently fixed.

**Sets.**
- Budget selection set: 5 NEW ei_2pop tasks, seeds 100-104, generated with
  `python -m bench.tasks configs/exp_ei_2pop.yaml --seeds 100 101 102 103 104 --out tasks/selection`
  and used for nothing else.
- Test set: the committed ei_2pop tasks `tasks/ei_2pop_K2_seed{1..9}.json` (9 tasks). Excluded:
  `ei_2pop_K2_seed0`, which was used in the Phase 0 smoke run.

**Budget rule (applied mechanically).** Run A, B-cma and C2-cma at total budgets 500, 1000 and
2000 on the selection set, 3 seeds each (seeds 0, 1, 2): 15 runs per (condition, budget). The
budget counts all evaluations, including the 3 reserved for the submission checks. A run
"succeeds" when its submitted parameters pass the final check. Pick the SMALLEST budget where
(1) A succeeds in at most 10% of its 15 runs (at most 1 run) AND (2) the best baseline (the larger
success count of B-cma and C2-cma) is not at 100% (not 15/15). If no budget qualifies, STOP and
report to Val; nothing else is run.

**Test conditions.** A, B-cma, B-tpe, C-cma (expert_v1), C2-cma, C2-tpe, all at the chosen
budget, 5 seeds (0-4) per task: 9 tasks x 5 seeds = 45 runs per condition.

**Metrics.** Success rate with Wilson 95% intervals; success by 25/50/100% of the budget (from the
checker-confirmed first success, as in `bench.batch`); wall-clock per run. A difference between
two conditions is only called a difference when their Wilson intervals do not overlap.

## ei_2pop budget selection: result (protocol above; selection set seeds 100-104, 3 seeds, 15 runs)
Successes: A 2/15, 2/15, 4/15 at budgets 500/1000/2000; B-cma 11/15, 10/15, 10/15; C2-cma 14/15 at
all three. Criterion 2 (best baseline not at 100%) holds at every budget; criterion 1 (A at most
1 of 15) fails at every budget, so NO BUDGET QUALIFIES. Per the protocol: stop and report to Val.
No test run was made. Data: `results/ei_selection/`.

## ei_2pop test protocol: AMENDMENT (written and committed BEFORE the test run)
Decided by Val after the budget selection found no budget under the original rule (A succeeded in
2/15, 2/15, 4/15 runs at 500/1000/2000, against a threshold of at most 1/15) and after expert_v1
was removed (`docs/decisions/0001-expert-objective.md`). This amendment replaces the "Test
conditions" and the budget of the protocol above; everything else (sets, metrics, "a difference
only when the Wilson intervals do not overlap", no change after seeing test results) stands.
- Budget: **500** total evaluations per run (Val's choice, an explicit exception to the budget rule;
  the rule's second criterion held at 500, the first failed by one run: A 2/15 against 1/15).
- Test set: `tasks/ei_2pop_K2_seed{1..9}.json` (9 tasks; `ei_2pop_K2_seed0` stays excluded).
- Conditions (C = optimizer + `expert_v2`, the only expert objective): **A, B-cma, B-tpe, C-cma,
  C-tpe**; 5 seeds (0-4) per task: 45 runs per condition, 225 in all.
- Config `configs/exp_ei_2pop.yaml` with `--budget 500`; all settings as in the protocol; the
  submission rule (reserve 3, check the top 3 distinct candidates) for every condition; A is random
  search ranked by `expert_v2`.
- Report: `docs/ei2pop_test.md` (condition table with Wilson intervals, success by 25/50/100% of the
  budget, wall-clock per run, per-task table). No change to code, settings or the report's
  definitions after seeing the results.
