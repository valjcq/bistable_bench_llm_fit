# Scripts

One-off analyses and run launchers. The library code is in `src/bench/`; these scripts only read its outputs or drive it. Each script explains itself in its header.

## Running the agent evaluations

These need a local llama.cpp server (see [docs/agent.md](../docs/agent.md)). Each one follows a protocol in [docs/NOTES.md](../docs/NOTES.md).

| script | what it does |
|---|---|
| `pilot.sh` | A few pilot episodes to check the setup. Pilot runs are excluded from every result. |
| `run_agent_eval.sh` | The first agent evaluation (prompt v1): pipeline P, then the main and pressure prompts on both sets. |
| `run_agent_eval_v2.sh` | The same evaluation with the v2 prompt (one added sentence about scores). |
| `run_agent_eval_ext.sh` | The extension: v1 and v2 on 18 new impossible tasks, 648 episodes. |
| `eval_progress.py` | Progress and ETA of a running evaluation, read from its run directories. |

## Analysing the agent runs

| script | produces |
|---|---|
| `agent_results.py` | [docs/agent_results.md](../docs/agent_results.md) and [docs/agent_results_v2.md](../docs/agent_results_v2.md) |
| `compare_agent_eval.py` | the v1 vs v2 tables in [docs/agent_v1_vs_v2.md](../docs/agent_v1_vs_v2.md) |
| `analyze_agent_ext.py` | the pre-registered extension analysis, [docs/agent_ext_analysis.md](../docs/agent_ext_analysis.md) |
| `show_episode.py` | one agent episode as readable Markdown, rebuilt from its run directory |
| `make_figures.py` | the figures in `docs/figures/`, from the committed result CSVs |

## Analysing the baselines

| script | produces |
|---|---|
| `phase0_diagnosis.py` | why expert v1 lost to the naive score, `results/phase0/diagnosis.md` |
| `selection_headroom.py` | the headroom analysis, `results/selection_headroom.md` (post-hoc) |
| `ei2pop_bootstrap.py` | task-level bootstrap of the ei_2pop test, `results/ei_test/bootstrap.md` |
| `task_bootstrap.py` | the same bootstrap for any test CSV (used for `results/wc_test/bootstrap.md`) |

## Building tasks and models

| script | what it does |
|---|---|
| `make_honesty_sets.py` | generates the honesty task sets, solvable and provably impossible tasks mixed |
| `wc_check.py` | recomputes Wilson–Cowan steady states from the paper's equations, to check the model card |
| `wc_feasibility.py` | how often random Wilson–Cowan draws give a bistable task, `results/wc_feasibility.md` |
| `wc_hysteresis_feasibility.py` | exploratory: how common hysteresis tasks would be, `results/wc_hysteresis_feasibility.md` (not adopted) |
