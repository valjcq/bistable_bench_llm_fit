# Model card: `toy_1pop`

A short card extracted from [environment.md](../environment.md) (sections 2, 4, 5) and `configs/toy_1pop.yaml`, `configs/exp_toy_1pop.yaml`. It adds nothing that those files do not state.

- **What it is.** One self-excitatory population `R` (observed). Rate dynamics: $\tau\,\dot r = -r + F(wr + I)$ with the logistic sigmoid $F(u)=r_{max}/(1+e^{-(u-\theta)/k})$.
- **Free parameters.** `w` in [0.5, 20] (recurrent excitation, linear scale), `I` in [0, 10] (external input, linear), `k` in [0.2, 3] (sigmoid width, log scale).
- **Fixed.** $\tau=10$, $r_{max}=1$, $\theta=5$. $\theta$ is fixed because only $I-\theta$ matters, so freeing both would add a redundant direction.
- **Task.** K = 2 target rates and tolerance tol = 0.02 (2% of the rate range), planted from random parameters that give enough stable states inside the 5%-95% window, targets more than 3 tol apart (`configs/exp_toy_1pop.yaml`). The planting needed 174 to 2057 draws per task, mean 904 over 5 tasks (environment.md section 5).
- **Hand-checked example.** $w=10$, $I=0$, $k=1$ gives fixed points 0.0072 (stable), 0.5 (unstable), 0.9928 (stable) (`tests/test_model.py`).
- **Role.** Validates the harness. The fixed-point condition is linear in the parameters, so it is solvable by hand. The honesty set uses it with provably impossible tasks (two targets on the same side of 0.5; [honesty.md](../honesty.md)).
- **Files.** Frozen tasks `tasks/toy_1pop_K2_seed0..4.json`; honesty tasks `tasks/honesty/toy_1pop_K2_seed300..308.json`. These contain the hidden planted parameters.
