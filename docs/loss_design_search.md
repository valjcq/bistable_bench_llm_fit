# Literature search brief: how are bistable network models fitted?

Brief for an online literature search (to be run by Claude with web search, or by hand). Context
and plan: NOTES.md, "Change of direction: can the agent design the loss?" (2026-10-15).

## Why this search

The next study asks whether an LLM agent can design a loss for fitting multistable network
models. Its baselines must be losses that researchers actually use, not only the two I wrote
(`naive`, `expert_v2`). I do not know the standard methods, so this search finds them.

## The problem, in neutral terms

- **Model.** Firing-rate network: tau_i dr_i/dt = -r_i + F_i(sum_j W_ij r_j + I_i), sigmoid F,
  a few populations. Weights are built from a few named parameters, with Dale's law (inhibitory
  outputs negative). Wilson-Cowan 1972 is one instance (with a refractory term).
- **Inverse problem.** Given K target rate vectors (usually K = 2: a low and a high state),
  find parameters so that each target is a *stable* fixed point within a tolerance. Some
  populations may be unobserved (no target for them).
- **Constraints.** A fixed budget of loss evaluations. Black-box optimizers (CMA-ES, TPE).
  Success is decided by an exact checker (fixed points plus Jacobian eigenvalues).
- **Current losses.** `naive`: simulate from a low and a high start, mean distance of the end
  states to the targets. `expert_v2`: for each target, the fixed-point residual
  |-r* + F(W r* + I)| plus a hinge penalty on the largest real part of the Jacobian's
  eigenvalues at the target.

## Questions to answer

1. How do people choose or fit parameters of rate models, neural mass models or small RNNs so
   that the system has prescribed stable states (bistability, multistability, working memory
   attractors)?
2. What objective does each method minimise? Does it simulate, use fixed points, use
   bifurcation structure, or use summary features?
3. Is `expert_v2` (residual plus spectral-abscissa hinge) a known construction? If so, under
   which name, and where is it published?
4. Is there prior work on LLMs designing loss or reward functions? This is related work for the
   agent side, not a baseline.
5. Are there benchmarks for fitting multistable dynamical systems?

## Where to look

Leads below come from memory and are **unverified**. Each must be confirmed by opening the
paper before it is cited anywhere.

| area | leads to check | why it may matter |
|---|---|---|
| Fixed points of RNNs | Sussillo & Barak 2013, "Opening the black box" (the "speed" q(x) = ½‖F(x)‖²) | may be the published form of the expert_v2 residual |
| Training attractor networks | FORCE learning (Sussillo & Abbott 2009), full-FORCE, low-rank RNNs (Mastrogiuseppe & Ostojic 2018) | how trained networks get their stable states |
| Stability-constrained optimisation | spectral abscissa minimisation (Vanbiervliet et al. 2009; Hennequin et al. 2014) | may be the published form of the stability term |
| Inverse bifurcation analysis | Lu, Engl & Schuster 2006, and systems-biology work on designing bistable switches (toggle switch, gene networks) | the same inverse problem in another field, possibly with standard losses |
| Simulation-based inference | Gonçalves et al. 2020 (eLife), the `sbi` toolbox | fits mechanistic models to summary features; bistability could be one feature |
| Feature-based and multi-objective fitting | Druckmann et al. 2007, BluePyOpt, NSGA-II | two constraints as two objectives instead of one weighted loss |
| Neural mass model fitting | Wilson-Cowan and Jansen-Rit fitting, Dynamic Causal Modelling, The Virtual Brain | how steady states are fitted in practice |
| Continuation tools | AUTO, MatCont, XPPAUT | design by tracking a branch of solutions, not by a loss: a non-loss baseline |
| LLMs writing objectives | Eureka (LLM-written reward functions, 2023), DiscoPOP (LLM-discovered losses, 2024), FunSearch, LLM-SR | precedent for the agent study |

Search terms to combine: "fitting bistable", "multistable", "attractor", "fixed point
constraint", "stability constraint", "spectral abscissa", "inverse bifurcation", "parameter
estimation", "rate model", "neural mass model", "Wilson-Cowan", "recurrent neural network",
"design of bistable switch", "loss function", "objective function", "LLM reward design",
"LLM discovered loss".

## How to search

1. Start with reviews and textbooks, then follow citations both ways (who cites the key papers).
2. Use primary sources. A method counts only once its paper has been opened and the objective
   read in it.
3. Search outside neuroscience too (systems biology, control theory), since the problem is
   generic.
4. Stop once new searches only return methods already listed.

## Output: `docs/loss_baselines.md`

One table row per method:

- reference with link;
- field;
- what is minimised, as an equation if possible;
- whether it needs simulation, fixed points, gradients or bifurcation tracking;
- how it handles stability;
- whether it fits the bench interface (one call = one evaluation, black-box optimiser);
- implementation effort.

Then:

- a ranked shortlist of 2-3 baselines, with the reason for each;
- what `expert_v2` corresponds to in the literature, if anything;
- the agent-side related work, separately;
- methods considered and rejected, with the reason.

## Rules

- Never cite a paper that was not opened. Mark anything read only in a summary or abstract as
  such.
- Never invent references, equations or results. If something is not found, say so.
- Keep "fitting to recorded data" apart from "designing parameters for prescribed states".
  This bench does the second.
