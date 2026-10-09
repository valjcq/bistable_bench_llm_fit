# Bistable Fit Bench

An environment and benchmark for fitting multistable neural models, built to test whether an LLM agent adds anything over a fixed expert optimization pipeline.

[![CI](https://github.com/valjcq/bistable_bench_llm_fit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/valjcq/bistable_bench_llm_fit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The question and the finding

I asked whether an LLM agent, given tools for simulating, checking and optimizing a small neural network model, can fit parameters so that the network holds given target states better than a fixed pipeline can. I built a deterministic referee first, then budget-matched baselines, then held-out tests, and only then the agent.

On these tasks the agent adds no solving power. One fixed setup (CMA-ES on an objective I designed, `expert_v2`) was the best or tied everywhere, so the measured room for an agent to improve on it ("selection headroom") is about zero. The local open model I tried (Qwen3.6-35B-A3B) solved every solvable task, but on provably impossible tasks it claimed a solution in 8 of 36 episodes, against 0 of 18 for the fixed pipeline. With 9 episodes per cell this is a signal, not a demonstrated difference: it does not reach the pre-registered criterion.

## The story in six steps

1. **Question.** Can an agent do the inverse problem (target rates in, parameters out) by orchestrating optimizers?
2. **Referee.** A deterministic environment with an exact fixed-point checker (Jacobian stability), a fixed-step simulator, three kept-apart scores and an evaluation budget. No LLM inside. [docs/environment.md](docs/environment.md)
3. **Baselines and objective design.** Random search, and CMA-ES / TPE on a naive or an expert score. My first expert score lost to plain simulation; classifying the failed runs found two traps, and the rewritten `expert_v2` was pre-registered before any run. [docs/decisions/0001-expert-objective.md](docs/decisions/0001-expert-objective.md)
4. **Held-out tests.** An E/I pair (`ei_2pop`) and the Wilson-Cowan 1972 model, with the budget chosen on separate selection tasks and the protocol written before the test run. [docs/protocol.md](docs/protocol.md)
5. **Headroom check.** Before building the agent I measured the gap between the best fixed setup and a per-task oracle. It was zero or near zero.
6. **Pivot to honesty.** The agent study asks instead whether the agent picks the right optimizer and whether it invents solutions on tasks with a proof that no solution exists. [docs/agent_study.md](docs/agent_study.md)

The longer version is [docs/STORY.md](docs/STORY.md).

## Key results

Every number below is from a file in this repository; [docs/results.md](docs/results.md) lists them all with their source files.

**Toy model, budget 200, 25 runs per condition (development tasks, expert v2 was designed after seeing them).** Source: `results/phase0/toy_grid_sr_summary.csv`. Before the final submission rule, B-cma was 18/25 and v2 with TPE 17/25 (`toy_grid_summary.csv`, `toy_grid_v2_summary.csv`).

| | CMA-ES | TPE |
|---|---|---|
| Random search | 0/25 | |
| Naive score | 17/25 | 12/25 |
| Expert v1 | 11/25 | 2/25 |
| Expert v2 | 24/25 | 18/25 |

**Held-out tests, 45 runs per condition.** Sources: `results/ei_test/test_summary.csv`, `results/wc_test/test_summary.csv`.

| test | random | naive+CMA | naive+TPE | expert v2+CMA | expert v2+TPE |
|---|---|---|---|---|---|
| ei_2pop, budget 500 | 8 | 36 | 31 | **44** | 35 |
| Wilson-Cowan 1972, budget 200 | 5 | 12 | 14 | **28** | 23 |

**Selection headroom (post-hoc).** Source: `results/selection_headroom.md`. Gap = per-task best configuration minus best single configuration.

| model | best single configuration | gap (plain) | gap (conservative) |
|---|---|---|---|
| toy, budget 200 | expert v2 + CMA-ES | 0.000 | 0.000 |
| ei_2pop, budget 500 | expert v2 + CMA-ES | 0.000 | 0.000 |
| Wilson-Cowan, budget 200 | expert v2 + CMA-ES | 0.067 | 0.000 |

**Agent honesty, local Qwen3.6-35B-A3B vs the fixed pipeline.** Source: `results/agent_eval/*.csv`; intervals in [docs/agent_results.md](docs/agent_results.md). 18 solvable and 9 impossible tasks per set; the pipeline was run once per set. "Pressure" adds one sentence about an upcoming experiment.

| set | prompt | solvable: solved | impossible: claimed "solution" | pipeline: claimed "solution" |
|---|---|---|---|---|
| toy | main | 18/18 | 3/9 | 0/9 |
| toy | pressure | 18/18 | 2/9 | 0/9 |
| Wilson-Cowan (as `model_C`) | main | 18/18 | 2/9 | 0/9 |
| Wilson-Cowan (as `model_C`) | pressure | 18/18 | 1/9 | 0/9 |

With the main prompt the agent started with CMA-ES on `expert_v2` in 20 of 27 episodes per set. The pressure prompt changed behaviour (more TPE, more overrides, more checks), not honesty.

## What this shows technically

- A **deterministic checker** with exact fixed points and Jacobian stability, cross-checked against a bit-reproducible RK4 simulator.
- **Budget-matched baselines** and one submission rule for all of them.
- **Pre-written protocols**: the protocol sections in [NOTES.md](NOTES.md) were committed before the runs they govern; changes are labelled amendments.
- **Held-out evaluation** with separate selection and test tasks, and reported failures of my own selection rules.
- A **leak-tested agent interface**: neutral model labels, renamed parameters, and tests that scan every agent-visible output.
- **Full JSONL logging** of every agent turn, with a replay tool.
- **Local LLM serving** through llama.cpp, with a preflight that checks the served model.
- **Reproducibility**: seeds, config hashes, frozen task files, `uv.lock`, git tags.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Reproduce a baseline table (about a minute; toy tasks, budget 200). It gives A 0/25, B-cma 17/25, B-tpe 12/25, C-cma 24/25, C-tpe 18/25, as in the table above (here `C` is `expert_v2`):

```bash
uv run python -m bench.batch configs/exp_toy_1pop.yaml --tasks "tasks/toy_1pop_*.json" \
  --conditions A B:cma B:tpe C:cma C:tpe --seeds 0-4 --jobs 4 --out /tmp/toy_grid --csv /tmp/toy_grid.csv
```

Run and replay one agent episode without an LLM (a scripted dry-run agent, so no GPU, server or key is needed):

```bash
uv run python -m bench.agent.run configs/exp_toy_1pop.yaml --agent configs/agent_local.yaml \
  --task tasks/honesty/toy_1pop_K2_seed300.json --seed 0 --dry-run --out /tmp/dry
uv run python scripts/show_episode.py /tmp/dry/toy_1pop_K2_seed300__dryrun__s0
```

The real Qwen episodes are not in the repository and the raw logs are not published for now. Their run directories are archived locally with checksums ([archives/MANIFEST.md](archives/MANIFEST.md)); the same `show_episode.py` replays them.

## Limitations

- One local, quantized model; no closed or larger model was run before publishing, so nothing here says how one behaves.
- 9 impossible episodes per cell, from 3 tasks x 3 seeds, so episodes are not independent. The 8 false claims are a signal, not a pre-registered result.
- Temperature 0.6: agent runs are not bit-reproducible. The transcripts and configs are the record.
- The tasks are easy to solve (few target numbers for many free parameters), so there is no selection headroom; this says little about harder fitting problems. The headroom analysis was post-hoc.
- The toy and Wilson-Cowan are the only models used for the agent.
- Expert v2 was designed on the toy development tasks; only the ei_2pop and Wilson-Cowan tests are held-out.
- **The committed task files contain the hidden planted parameters** (and the impossibility flags). They are needed to run the checker. A model trained on this repository could have seen the answers, so new tasks must be generated for any future evaluation.
- The post-hoc classification of the 8 claimed solutions is descriptive and small-sample ([docs/agent_study.md](docs/agent_study.md)).

## How I used AI coding assistants

I built the code with an AI coding assistant (Claude Code) in reviewed checkpoints: each stage was reviewed before the next one, and the review findings are recorded in [NOTES.md](NOTES.md). I can explain and defend each part of the code and the design.

## Repo map

| path | contents |
|---|---|
| `src/bench/` | the deterministic environment (models, checker, simulator, scoring, budget, tasks), optimizers, tool layer, logging; `src/bench/agent/` is the LLM loop and backends, kept apart from the environment |
| `tests/` | unit, golden-file, leak and determinism tests; no test needs a GPU, a server or `runs/` |
| `configs/`, `prompts/` | experiment and model configs, agent configs, system prompts |
| `tasks/` | frozen task files (with hidden planted parameters), selection sets and honesty sets |
| `results/` | small committed result summaries (CSV and markdown) |
| `docs/` | [STORY](docs/STORY.md), [design decisions](docs/design_decisions.md), [protocols](docs/protocol.md), [results](docs/results.md), [agent study](docs/agent_study.md), [environment](docs/environment.md), [model cards](docs/model_cards/) |
| `archives/` | pointers and checksums for the raw agent logs; no bulk data |
| `NOTES.md` | the dated decision log and the protocols |

Tags: `v0.1-env` (the deterministic environment) and `phase0-expert-v1` (the last commit that still contains the first expert objective, `expert_v1`; it also contains the ei_2pop test protocol and its budget-selection runs).

## Cite and contact

Valentin Jacquin, computational neuroscience graduate working on ML development. Contact: https://github.com/valjcq. Cite with [CITATION.cff](CITATION.cff). Licensed under MIT ([LICENSE](LICENSE)).
