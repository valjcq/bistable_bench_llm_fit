## (a) Final best expert score of FAILED C runs

| condition | failed runs | < tol | [tol, 0.5) | [0.5, 0.6) | [0.6, 0.75) | [0.75, 1.0) | >= 1.0 |
|---|---|---|---|---|---|---|---|
| C-cma | 14 | 0 | 0 | 3 | 3 | 2 | 6 |
| C-tpe | 23 | 0 | 0 | 4 | 1 | 3 | 15 |

C-cma: median 0.809, min 0.517, max 1.141; targets matched at the best point (checker): {0: 6, 1: 8} (key = number of targets matched).

C-tpe: median 1.027, min 0.513, max 1.141; targets matched at the best point (checker): {0: 15, 1: 8} (key = number of targets matched).

### Plateau level (1 + separation) / 2

- C-cma: 5 of 14 failed runs end within 0.01 of their task's (1 + separation) / 2.
- C-tpe: 4 of 23 failed runs end within 0.01 of their task's (1 + separation) / 2.

## (b) What the evaluated points are (checker, off-budget), 25 runs x 200 evals

| condition | monostable (1 stable state) | no stable state | bistable+ (>=2 stable) | success |
|---|---|---|---|---|
| B-cma | 0.498 | 0.000 | 0.502 | 0.040 |
| C-cma | 0.532 | 0.000 | 0.468 | 0.024 |

## (c) Best-so-far expert score of stuck C-cma runs

| run | @1 | @10 | @25 | @50 | @100 | @150 | @200 | last improvement | monostable evals | evals with score >= 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| task0 seed0 | 1.505 | 1.128 | 1.128 | 0.722 | 0.722 | 0.722 | 0.722 | 38 | 172/200 | 200/200 |
| task3 seed0 | 1.481 | 1.118 | 0.809 | 0.809 | 0.809 | 0.809 | 0.809 | 25 | 57/200 | 200/200 |

## Per-task descriptors and success (of 5 seeds)

| task | low target | high target | low-edge dist | high-edge dist | separation | planted margin_eig | planted margin_dist | A | B-cma | B-tpe | C-cma | C-tpe | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed0 | 0.273 | 0.718 | 0.223 | 0.232 | 0.445 | 0.0129 | 0.207 | 0 | 3 | 0 | 0 | 0 | 0 |
| seed1 | 0.177 | 0.949 | 0.127 | 0.001 | 0.771 | 0.0158 | 0.100 | 0 | 5 | 4 | 4 | 0 | 3 |
| seed2 | 0.101 | 0.920 | 0.051 | 0.030 | 0.819 | 0.0487 | 0.358 | 0 | 5 | 4 | 5 | 0 | 2 |
| seed3 | 0.210 | 0.829 | 0.160 | 0.121 | 0.619 | 0.0222 | 0.238 | 0 | 1 | 1 | 0 | 0 | 0 |
| seed4 | 0.052 | 0.851 | 0.002 | 0.099 | 0.799 | 0.0260 | 0.164 | 0 | 4 | 3 | 2 | 2 | 0 |

edge dist = distance of the target to the window edge (0.05 / 0.95); separation = high - low (the guard is 3 tol = 0.06); margins are those of the planted solution (`margin_eig` = slowest decay rate, `margin_dist` = distance to the unstable point).
