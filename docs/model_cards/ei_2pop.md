# Model card: `ei_2pop`

A short card extracted from [environment.md](../environment.md) (sections 4, 5) and `configs/ei_2pop.yaml`, `configs/exp_ei_2pop.yaml`. It adds nothing that those files do not state.

- **What it is.** An excitatory (E) and an inhibitory (I) population, Wilson-Cowan-style, with generic bounds not taken from any paper. Same logistic rate dynamics as the toy, with $W = \begin{pmatrix} +w_{EE} & -w_{EI} \\ +w_{IE} & -w_{II}\end{pmatrix}$: parameters are stored positive and the builder applies the signs (Dale's law).
- **Partial observation.** Targets constrain E only (`observed: [E]`); I is free.
- **Free parameters (6, all linear scale).** `w_EE`, `w_EI`, `w_IE`, `w_II` in [0, 20]; `I_E`, `I_I` in [0, 10].
- **Fixed.** $\tau_E=10$, $\tau_I=5$ (unequal on purpose), $\theta_E=\theta_I=5$, $k_E=k_I=1$, $r_{max}=1$ for both. The time constants matter for stability: with slow inhibition an E/I pair can oscillate forever instead of settling.
- **Task.** K = 2 target E-rates, tol = 0.02, same planting window and separation as the toy. Planting needed 4 to 1227 draws per task, mean 264 over 10 tasks (environment.md section 5).
- **Role.** Held-out test with budget 500 ([results.md](../results.md) section 2). The configuration poses few target numbers for many free parameters (2 target rates, 6 free parameters).
- **Files.** Frozen tasks `tasks/ei_2pop_K2_seed0..9.json` (seeds 1-9 form the test set, seed 0 was used in a smoke run), selection tasks `tasks/selection/ei_2pop_K2_seed100..104.json`. These contain the hidden planted parameters.
