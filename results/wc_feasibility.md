Wilson-Cowan feasibility: 5000 draws in the wc1972 box (seed 0), P ~ U[-0.3, 0.3], |P| >= 0.05; tol = 0.01, planting window E in [0.025, 0.475], separation >= 0.03.

| criterion | prefilter (16 starts): fraction [Wilson 95%] | full (256 starts): fraction [Wilson 95%] |
|---|---|---|
| >= 2 stable states | 1852/5000 = 0.3704 [0.3571, 0.3839] | 2147/5000 = 0.4294 [0.4157, 0.4432] |
| >= 3 stable states | 4/5000 = 0.0008 [0.0003, 0.0021] | 4/5000 = 0.0008 [0.0003, 0.0021] |
| >= 2 usable stable states (E in window) | 1/5000 = 0.0002 [0.0000, 0.0011] | 1/5000 = 0.0002 [0.0000, 0.0011] |
| K=2: >= 2 usable, separated by >= 3 tol | 1/5000 = 0.0002 [0.0000, 0.0011] | 1/5000 = 0.0002 [0.0000, 0.0011] |
| K=3: >= 3 usable, separated by >= 3 tol | 0/5000 = 0.0000 [0.0000, 0.0008] | 0/5000 = 0.0000 [0.0000, 0.0008] |

Draws per task (1 / fraction; range from the Wilson interval):

| K | prefilter (what planting uses) | full |
|---|---|---|
| 2 | 5000 [883, 28325] | 5000 [883, 28325] |
| 3 | no success in the sample | no success in the sample |

Where the stable states sit (2147 draws with >= 2 stable states, full effort):

- lowest stable E, percentiles [5, 25, 50, 75, 95]: [-0.0018, -0.0, -0.0, 0.0, 0.0011]
- highest stable E, percentiles [5, 25, 50, 75, 95]: [0.4663, 0.4983, 0.5, 0.5, 0.5]
- lowest state inside the window (E >= 0.025): 0.0005 of them; highest inside (E <= 0.475): 0.0592; both: 0.0005

Sensitivity of the K=2 count (of all draws) to the window, same separation rule:

| window for E | draws with K=2 |
|---|---|
| [0.025, 0.475] | 1/5000 |
| [0.005, 0.495] | 35/5000 |
| [0.000, 0.500] | 1072/5000 |
