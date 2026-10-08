EXPLORATORY: hysteresis feasibility for Wilson-Cowan (not adopted, no tasks planted).

2000 random draws in the wc1972 box (seed 0), Q = 0; P swept over [-0.5, 1.0] in steps of 0.01; stable states found with the planting-level checker (16 starts + corners). An interval = maximal run of consecutive P values with >= 2 stable states; width = last P - first P.

- draws with at least one bistable interval: 1114/2000 = 0.557
- number of intervals found: 1416 (draws with 2 or more intervals: 99)
- interval width, all intervals, percentiles [5, 25, 50, 75, 95]: [0.0, 0.22, 1.26, 1.5, 1.5]; mean 0.94; widest interval per draw: [0.15, 0.89, 1.5, 1.5, 1.5]
- draws with an interval inside [-0.3, 0.6] and width in [0.1, 0.5]: 34/2000 = 0.017 (37 such intervals)

Active state at the interval midpoint (highest stable E there; window [0.05, 0.45]):

| intervals | count | active E in window | in window and >= 3 tol above the lowest |
|---|---|---|---|
| all | 1416 | 136 (0.096) | 136 (0.096) |
| inside [-0.3, 0.6], width in [0.1, 0.5] | 37 | 13 (0.351) | 13 (0.351) |

- draws with a qualifying interval whose midpoint has a usable active state: 13/2000 = 0.0065

Effort check: on the first 100 draws, at every 10th P value the 256-start search found bistability where the 16-start search did not in 12 of 1600 cases.
