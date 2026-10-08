Wilson-Cowan feasibility, K=2: 5000 draws per scenario (seed 0), P ~ U[-0.3, 0.3], |P| >= 0.05, tol = 0.01.
Asymmetric rule: high target E in [0.05, 0.45], low target free, separation >= 0.03. Symmetric (old) rule: both in [0.025, 0.475].

### box of wc1972.yaml (adopted)

| criterion | prefilter (planting) | full (256 starts) |
|---|---|---|
| >= 2 stable states | 2096/5000 = 0.4192 [0.4056, 0.4329] | 2152/5000 = 0.4304 [0.4167, 0.4442] |
| K=2, asymmetric rule (adopted) | 68/5000 = 0.0136 [0.0107, 0.0172] | 74/5000 = 0.0148 [0.0118, 0.0185] |
| K=2, symmetric rule (old) | 1/5000 = 0.0002 [0.0000, 0.0011] | 1/5000 = 0.0002 [0.0000, 0.0011] |

Draws per task, K=2, asymmetric rule: **74 [58, 93]** with the planting prefilter, 68 [54, 85] with the full search (1 / fraction, [range from the Wilson interval]).

Stable states of the 2152 draws with >= 2 (full): lowest E percentiles [5, 25, 50, 75, 95]: [-0.0017, -0.0, -0.0, 0.0, 0.0011]; highest E: [0.4663, 0.4983, 0.5, 0.5, 0.5].

### a_e in [1, 2] (comparison only, not adopted)

| criterion | prefilter (planting) | full (256 starts) |
|---|---|---|
| >= 2 stable states | 1781/5000 = 0.3562 [0.3430, 0.3696] | 1831/5000 = 0.3662 [0.3530, 0.3797] |
| K=2, asymmetric rule (adopted) | 303/5000 = 0.0606 [0.0543, 0.0676] | 332/5000 = 0.0664 [0.0598, 0.0736] |
| K=2, symmetric rule (old) | 13/5000 = 0.0026 [0.0015, 0.0044] | 14/5000 = 0.0028 [0.0017, 0.0047] |

Draws per task, K=2, asymmetric rule: **17 [15, 18]** with the planting prefilter, 15 [14, 17] with the full search (1 / fraction, [range from the Wilson interval]).

Stable states of the 1831 draws with >= 2 (full): lowest E percentiles [5, 25, 50, 75, 95]: [-0.0243, -0.0026, -0.0002, 0.0011, 0.0089]; highest E: [0.3392, 0.468, 0.4904, 0.4973, 0.4996].
