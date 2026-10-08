"""A fake backend that replays scripted turns, for tests (no LLM)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from bench.agent.loop import AssistantTurn, ToolCall

Messages = list[dict[str, Any]]


def turn(*calls: tuple[str, dict[str, Any]], text: str = "") -> AssistantTurn:
    """An assistant turn with tool calls given as (name, arguments); ids are c1, c2, ..."""
    return AssistantTurn(
        text=text, tool_calls=[ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls, start=1)]
    )


def last_tool_result(messages: Messages, index: int = -1) -> dict[str, Any]:
    """The JSON result of the last (or `index`-th) tool message, as the agent saw it."""
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    return results[index]


class FakeBackend:
    """Replays `script`: each item is an AssistantTurn or a function (messages) -> AssistantTurn,
    so a scripted agent can use what the tools returned (e.g. the best parameters of a run).
    Running out of script is an error (a harness error in the loop), as is a function that raises.
    """

    def __init__(self, script: list[AssistantTurn | Callable[[Messages], AssistantTurn]]) -> None:
        self._script = list(script)
        self.calls: list[tuple[Messages, list[dict[str, Any]]]] = []

    def generate(self, messages: Messages, tools: list[dict[str, Any]]) -> AssistantTurn:
        self.calls.append(([dict(m) for m in messages], tools))
        if len(self.calls) > len(self._script):
            raise RuntimeError("the fake backend script is exhausted")
        item = self._script[len(self.calls) - 1]
        return item(messages) if callable(item) else item


def reference_backend() -> FakeBackend:
    """A scripted policy for --dry-run: get_task, one run of cma on expert_v2 with the whole
    budget but one evaluation, a check of the best parameters, then submit the solution if the
    checker passed, else no_solution (the reference pipeline P expressed as turns)."""

    def run(messages: Messages) -> AssistantTurn:
        budget = last_tool_result(messages)["budget"]["total"]
        return turn(
            ("run_optimizer", {"name": "cma", "sim_budget": budget - 1, "objective": "expert_v2"})
        )

    def check(messages: Messages) -> AssistantTurn:
        return turn(("check_bistability", {"params": last_tool_result(messages)["best_params"]}))

    def submit(messages: Messages) -> AssistantTurn:
        checked, run_result = last_tool_result(messages), last_tool_result(messages, -2)
        if checked.get("success"):
            return turn(("submit", {"outcome": "solution", "params": run_result["best_params"]}))
        return turn(("submit", {"outcome": "no_solution"}))

    return FakeBackend([turn(("get_task", {})), run, check, submit])
