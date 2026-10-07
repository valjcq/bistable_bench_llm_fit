"""The tool layer: everything an agent can do in an episode, as JSON-in / JSON-out functions.

An `Episode` owns one Environment, the optimizer-run registry and the "submitted" flag. An agent
(scripted now, an LLM later) talks to it only through `Episode.call(name, arguments)`, which

  1. validates the arguments against the tool's JSON Schema,
  2. runs the tool,
  3. turns every failure into {"error": {"type": ..., "message": ...}} (nothing is raised),
  4. logs the call (with an `invalid` flag for bad calls) when a logger is attached.

Invalid calls cost nothing: validation happens before any budgeted work. After `submit`, every
tool returns an "episode_over" error. The planted parameters are never reachable from here: the
Episode is built with `Environment.from_task`, which reads only the targets and the seed.

Error types: "invalid_argument" (bad arguments, free), "budget_exhausted" (nothing left to
spend), "episode_over" (already submitted), "unknown_tool". Only these expected failures are
caught; an unexpected exception is a bug and propagates.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from bench.budget import BudgetExhausted
from bench.config import ExperimentConfig
from bench.env import OBJECTIVES, Environment
from bench.logging import JsonlLogger
from bench.optimizer_runs import RunRegistry
from bench.optimizers import OPTIMIZERS
from bench.optimizers.base import RunResult
from bench.schema import validate
from bench.tasks import Task

RUN_FIELDS = tuple(RunResult.__dataclass_fields__)  # what get_run_results can return
DEFAULT_RUN_FIELDS = [f for f in RUN_FIELDS if f != "history"]  # history can be long


JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"  # what the schemas declare


def _error(type_: str, message: str) -> dict[str, Any]:
    return {"error": {"type": type_, "message": message}}


class Episode:
    """One agent episode: environment + runs + submitted flag, behind `call`."""

    def __init__(
        self,
        config: ExperimentConfig,
        task: Task,
        seed: int,
        logger: JsonlLogger | None = None,
        role: str = "agent",
    ) -> None:
        self.logger = logger
        self.role = role  # who is calling the tools (an LLM later); logged on every line
        if logger is not None:
            logger.role = role
        self.env = Environment.from_task(
            config, task, event_hook=logger.env_hook if logger else None
        )
        self.runs = RunRegistry(self.env, episode_seed=seed)
        self.submitted = False
        self.final_metric: dict[str, Any] | None = None  # set by submit, for the harness

    # ---- schemas -----------------------------------------------------------------------

    def tool_specs(self) -> list[dict[str, Any]]:
        """Name, description and JSON Schema (`input_schema`) of every tool.

        Provider-neutral: the LLM backends convert these to their own tool format. Parameter
        names and bounds are this task's, so a bad parameter dict is caught by the schema.
        """
        free = self.env.space.free
        params = {
            "type": "object",
            "description": "Values of the free parameters (see get_task), within their bounds.",
            "properties": {
                p.name: {"type": "number", "minimum": p.low, "maximum": p.high} for p in free
            },
            "required": [p.name for p in free],
            "additionalProperties": False,
        }
        n_pops = len(self.env.describe()["populations"])
        budget = self.env.budget_total
        specs = [
            {
                "name": "get_task",
                "description": "The task: target states, tolerance, free parameters with bounds "
                "and log-scale flags, populations, and the evaluation budget. Free.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "list_optimizers",
                "description": "Available optimizers with their hyperparameters (type, range, "
                "default). Free.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "run_optimizer",
                "description": "Run an optimizer on the unit cube of the free parameters for at "
                "most sim_budget evaluations of the chosen objective ('naive': simulate and "
                "compare end states; 'expert': checker-based). Costs one evaluation per "
                "objective call, taken from the global budget. Returns a run_id and a summary.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": sorted(OPTIMIZERS)},
                        "hyperparameters": {
                            "type": "object",
                            "description": "Overrides of the defaults in list_optimizers.",
                        },
                        "sim_budget": {"type": "integer", "minimum": 1, "maximum": budget},
                        "objective": {"type": "string", "enum": list(OBJECTIVES)},
                        # optional starting point: a parameter dict, or null for none
                        "init": {**params, "type": ["object", "null"]},
                        "seed": {"type": ["integer", "null"]},
                    },
                    "required": ["name", "sim_budget", "objective"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_run_results",
                "description": "Stored details of a past optimizer run. Free.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(RUN_FIELDS)},
                            "description": f"Default: all but 'history'. Options: {RUN_FIELDS}",
                        },
                    },
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "check_bistability",
                "description": "Full checker report for a parameter set: all fixed points with "
                "stability, which targets are matched, success flag, margins. Costs 1 evaluation.",
                "input_schema": {
                    "type": "object",
                    "properties": {"params": params},
                    "required": ["params"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "simulate",
                "description": "Simulate one trajectory from init_state (all populations) for "
                "time T and report the end state and whether it settled. Costs 1 evaluation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "params": params,
                        "init_state": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": n_pops,
                            "maxItems": n_pops,
                        },
                        "T": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "required": ["params", "init_state", "T"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_budget",
                "description": "Evaluations used and remaining. Free.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "submit",
                "description": "Submit final parameters and end the episode. Returns the final "
                "evaluation metric. Free, but only the first valid submission counts.",
                "input_schema": {
                    "type": "object",
                    "properties": {"params": params},
                    "required": ["params"],
                    "additionalProperties": False,
                },
            },
        ]
        for spec in specs:
            spec["input_schema"]["$schema"] = JSON_SCHEMA_DRAFT
        return specs

    # ---- the single entry point -------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run tool `name` with `arguments`; always returns a JSON-serializable dict."""
        arguments = {} if arguments is None else arguments
        invalid = False
        call_id = self.logger.begin_call() if self.logger is not None else None
        specs = {s["name"]: s for s in self.tool_specs()}
        if self.submitted:
            out, invalid = _error("episode_over", "the episode has ended (submit was called)"), True
        elif name not in specs:
            out, invalid = _error("unknown_tool", f"unknown tool {name!r}"), True
        elif problems := (
            validate(arguments, specs[name]["input_schema"])
            if isinstance(arguments, dict)
            else ["$: arguments must be an object"]
        ):
            out, invalid = _error("invalid_argument", "; ".join(problems)), True
        else:
            try:
                out = getattr(self, f"_tool_{name}")(**arguments)
            except BudgetExhausted as e:
                out = _error("budget_exhausted", str(e))
            except ValueError as e:  # raised by the environment BEFORE charging anything
                out, invalid = _error("invalid_argument", str(e)), True
            finally:
                if self.logger is not None:
                    self.logger.end_call()  # (also on a bug: the next line must not be a child)
        out = json.loads(json.dumps(out))  # guarantees plain JSON types for the caller
        if self.logger is not None:
            self.logger.end_call()
            self.logger.log(
                self.role,
                name,
                arguments,
                out,
                self.env.budget_remaining,
                invalid=invalid,
                call_id=call_id,
            )
        return out

    # ---- the tools (arguments already validated against the schema) ------------------

    def _tool_get_task(self) -> dict[str, Any]:
        d = self.env.describe()
        d.pop("config_hash")
        d["budget"] = {"total": d["budget"], "remaining": self.env.budget_remaining}
        return d

    def _tool_list_optimizers(self) -> dict[str, Any]:
        return {"optimizers": [s.to_dict() for s in self.runs.specs()]}

    def _tool_run_optimizer(
        self,
        name: str,
        sim_budget: int,
        objective: str,
        hyperparameters: dict[str, Any] | None = None,
        init: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        # The optimizer's own evaluations are logged with role "optimizer", not the agent's.
        if self.logger is not None:
            self.logger.role = "optimizer"
        try:
            r = self.runs.run(name, objective, sim_budget, hyperparameters, init, seed)
        finally:
            if self.logger is not None:
                self.logger.role = self.role
        return {
            "run_id": r.run_id,
            "optimizer": r.optimizer,
            "best_score": r.best_score,
            "best_params": r.best_params,
            "evals_used": r.evals_used,
            "converged": r.converged,
            "stopped_reason": r.stopped_reason,
            "budget_remaining": self.env.budget_remaining,
        }

    def _tool_get_run_results(self, run_id: str, fields: list[str] | None = None) -> dict:
        d = self.runs.get(run_id).to_dict()
        return {"run_id": run_id, **{f: d[f] for f in (fields or DEFAULT_RUN_FIELDS)}}

    def _tool_check_bistability(self, params: dict[str, float]) -> dict[str, Any]:
        return self.env.check(params).to_dict()

    def _tool_simulate(self, params: dict, init_state: list[float], T: float) -> dict:
        return asdict(self.env.simulate(params, init_state, T))

    def _tool_get_budget(self) -> dict[str, int]:
        e = self.env
        return {"total": e.budget_total, "used": e.n_evaluations, "remaining": e.budget_remaining}

    def _tool_submit(self, params: dict[str, float]) -> dict[str, Any]:
        metric = asdict(self.env.final_report(params))
        self.submitted = True
        self.final_metric = metric
        return {"evaluation_metric": metric, "evals_used": self.env.n_evaluations}
