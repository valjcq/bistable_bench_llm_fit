# 0001. The expert objective: remove v1, keep v2

Status: accepted (Val's decision after the Phase 0 review and the ei_2pop budget selection).
Date: 2026-10-07.

## Decision

The environment has two objectives: `naive` and `expert_v2` (the name is kept; old logs say
`expert` for v1). The first expert objective, `expert_v1`, is removed from the code, tests,
config and conditions. Condition C now means "optimizer + `expert_v2`". Everything Phase 0
produced with v1 is reproducible at the git tag **`phase0-expert-v1`**
(`git checkout phase0-expert-v1`).

## What v1 was

Built on the fixed-point checker. For each target: if matched (a distinct stable fixed point
within tol), it adds the real error (at most tol); if unmatched, it adds the distance to the
nearest fixed point of any kind plus a constant penalty of 1.0. The score is the mean over
targets. So "all targets matched" gives at most tol, and each unmatched target adds at least
penalty / K = 0.5.

## v1's traps

- **Saddle on a target.** A target that coincides with an unstable fixed point counts as
  unmatched and is charged the distance to the nearest fixed point (which can be that very
  saddle, distance about 0) plus the full penalty. The score carries no information about how
  unstable that state is or how to make it stable.
- **Cliff at tol.** The cost of a target jumps from at most tol to at least 1.0 when its error
  crosses tol. A stable state a hair outside tolerance is scored like a missing one, so there is
  no gradient toward the last bit of accuracy.
- A third effect, a **plateau**: a monostable state sitting exactly on one target gives the
  constant score (1 + separation) / 2 whatever the parameters do.

## Evidence (Phase 0, toy, budget 200, 5 tasks x 5 seeds; details in `docs/phase0.md`)

- With the same optimizer, the naive objective beat v1: B-cma 18/25 against C-cma 11/25; B-tpe
  12/25 against C-tpe 2/25.
- Classification of the final best parameters of the failed v1 runs (priority a > b > c > d;
  non-exclusive counts in brackets):

  | condition | failed | (a) monostable plateau | (b) near miss, stable state within 3 tol of an unmatched target | (c) nearest fixed point to an unmatched target is unstable | (d) other |
  |---|---|---|---|---|---|
  | C-cma | 14 | 5 (5) | 4 (4) | 5 (6) | 0 (0) |
  | C-tpe | 23 | 4 (4) | 7 (7) | 10 (15) | 2 (2) |

  Classes (c) and (b) are the failures the saddle trap and the cliff predict; together they are
  9 of 14 and 17 of 23 failed runs. The plateau is a minority. This is descriptive evidence about
  where v1 runs end, not a proof of cause.

## v2: the definition (pre-registered in commit 784d4a0, before any v2 run)

For each target k:
- complete the target state r*: observed populations take the target values; each unobserved
  population solves its own fixed-point equation given the others (for ei_2pop:
  r_I = F_I(w_IE E* - w_II r_I + I_I), unique on [0, r_max], solved with `brentq`);
- `residual_k` = max over observed populations of |-r* + F(W r* + I)|;
- `stab_k` = max(0, max Re eig(J(r*)) * max(tau) + delta), delta = 0.01;
- score = mean over targets of (`residual_k` + lambda * `stab_k`), lambda = 1.0.

No fixed-point search and no simulation: one call is one evaluation. Success is decided only by
the checker. It supports at most one unobserved population (the only case in the shipped
models). The score is zero exactly when every target is a stable fixed point with margin delta,
and it is continuous in the parameters (no tolerance, no matching), which removes both traps.
delta and lambda were not tuned.

## Toy development results (the toy is the development set)

| condition | old submission rule | new rule (reserve 3, check top 3) |
|---|---|---|
| C-cma with v1 | 11/25 | 11/25 |
| C-cma with v2 | 24/25 | 24/25 |
| C-tpe with v1 | 2/25 | 2/25 |
| C-tpe with v2 | 17/25 | 18/25 |
| B-cma (naive) | 18/25 | 17/25 |

v2 was designed after seeing these toy tasks, so these are development numbers, not a clean
test. The best v2 score is not always a checker success (one run of C-tpe-v2 evaluated a
successful point but submitted another), which is why every baseline uses the submission rule
of NOTES.md.

## Held-out ei_2pop test

Pending when this decision was written. Protocol (with its amendment) is in NOTES.md; results
will be in `docs/ei2pop_test.md`. The budget selection under the original rule found no budget
where random search succeeded in at most 10% of runs (2/15, 2/15, 4/15 at 500/1000/2000); the
budget 500 was chosen by Val anyway, as an explicit exception recorded in the amendment.

## Consequences

- Phase 0 numbers that say "expert" or C (v1) belong to the tag, and `results/phase0/` is
  unchanged. Condition A in those results was random search ranked by v1; from now on it is
  ranked by v2.
- The penalty / K > tol config check and the `expert_penalty` setting no longer exist, because
  they only served v1.
