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
