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
