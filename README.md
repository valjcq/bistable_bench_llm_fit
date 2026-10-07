# Bistable Fit Bench

A benchmark that asks: can an LLM agent autonomously fit multistable neural rate models by
orchestrating numerical optimizers through tool calls? This repository holds the deterministic
environment the agent will work in: rate-network models (a 1-population toy and an E/I pair), an
exact/multi-start fixed-point checker, a fixed-step simulator, three kept-apart scoring functions
(naive, expert, final metric), an evaluation budget, and planted tasks whose solutions are known to
exist. Nothing here depends on an LLM or an agent.

How the environment works: [docs/environment.md](docs/environment.md).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # creates .venv from uv.lock
```

## Tests and lint

```bash
uv run pytest
uv run ruff check .
```

## Status

- Done: deterministic environment (models, checker, simulator, scoring, budget, task generation,
  configs, saved task files in `tasks/`). See `NOTES.md` for design decisions and known caveats.
- Next: optimizers, then the agent.

Tasks are generated once with `python -m bench.tasks configs/exp_ei_2pop.yaml --seeds 0 1 2`
and saved as JSON (they include the hidden planted parameters: never show them to an agent).
