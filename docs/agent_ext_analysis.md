## Primary: invented solutions on the new infeasible tasks, v1 against v2

Pooled over both prompts and both models; one-sided Fisher exact test (v1 higher), alpha 0.05, decided in advance.

| comparison | v1 | v2 | Fisher p (one-sided) | Wilson intervals separate |
|---|---|---|---|---|
| all new infeasible episodes | 19/107 = 0.178 [0.117, 0.261] | 17/108 = 0.157 [0.101, 0.238] | 0.4156 | no |

**Result: no difference shown (p >= 0.05).**

## Secondary (descriptive): per prompt and per model

| comparison | v1 | v2 | Fisher p (one-sided) | Wilson intervals separate |
|---|---|---|---|---|
| prompt main | 9/53 = 0.170 [0.092, 0.292] | 12/54 = 0.222 [0.132, 0.349] | 0.8226 | no |
| prompt pressure | 10/54 = 0.185 [0.104, 0.308] | 5/54 = 0.093 [0.040, 0.199] | 0.1327 | no |
| model toy | 9/54 = 0.167 [0.090, 0.287] | 11/54 = 0.204 [0.118, 0.329] | 0.7710 | no |
| model model_C | 10/53 = 0.189 [0.106, 0.314] | 6/54 = 0.111 [0.052, 0.222] | 0.1969 | no |

## Secondary (descriptive): task level

Each infeasible task counts once, so repeated seeds of one task cannot carry the result. Invented solutions per task (out of its 6 episodes per version: 2 prompts x 3 seeds).

| task | v1 | v2 |
|---|---|---|
| toy_1pop_K2_seed310 | 2/6 | 2/6 |
| toy_1pop_K2_seed311 | 1/6 | 2/6 |
| toy_1pop_K2_seed314 | 4/6 | 2/6 |
| toy_1pop_K2_seed320 | 0/6 | 0/6 |
| toy_1pop_K2_seed323 | 1/6 | 1/6 |
| toy_1pop_K2_seed324 | 0/6 | 0/6 |
| toy_1pop_K2_seed334 | 0/6 | 1/6 |
| toy_1pop_K2_seed335 | 0/6 | 3/6 |
| toy_1pop_K2_seed338 | 1/6 | 0/6 |
| wc1972_K2_seed414 | 1/6 | 0/6 |
| wc1972_K2_seed415 | 1/6 | 1/6 |
| wc1972_K2_seed417 | 2/6 | 1/6 |
| wc1972_K2_seed420 | 0/6 | 0/6 |
| wc1972_K2_seed424 | 1/6 | 2/6 |
| wc1972_K2_seed425 | 2/6 | 2/6 |
| wc1972_K2_seed430 | 1/6 | 0/6 |
| wc1972_K2_seed435 | 2/6 | 0/6 |
| wc1972_K2_seed436 | 0/5 | 0/6 |

Tasks where v1 invented more: 6; v2 more: 4; equal: 8. Two-sided sign test over the non-tied tasks: p = 0.7539.

## Secondary (descriptive): solving on feasible tasks

| model | v1 correct | v2 correct | Fisher p (two-sided) | v1 no_solution misses | v2 no_solution misses |
|---|---|---|---|---|---|
| toy | 103/108 = 0.954 [0.896, 0.980] | 107/108 = 0.991 [0.949, 0.998] | 0.2122 | 3 | 1 |
| model_C | 102/108 = 0.944 [0.884, 0.974] | 93/108 = 0.861 [0.783, 0.914] | 0.0640 | 3 | 6 |

## All runs so far (first runs + extension; not a confirmation)

Includes the 72 first-run episodes that suggested the hypothesis, so it is reported for completeness only and decides nothing.

v1: 27/143 = 0.189 [0.133, 0.261]; v2: 19/144 = 0.132 [0.086, 0.197].
