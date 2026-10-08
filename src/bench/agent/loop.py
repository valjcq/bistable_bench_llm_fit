"""The agent loop: a provider-neutral conversation between a backend (an LLM, or a fake) and an
Episode.

A `Backend` takes the conversation so far (provider-neutral messages) and the tool specs and
returns ONE assistant turn: text and/or tool calls. The loop executes the tool calls through
`Episode.call`, feeds the results back as tool messages and repeats, until

    submit        the agent submitted                           (stopped_reason "submit")
    cap           the tool-call cap is reached without submit    ("tool_call_cap")
    max turns     the turn limit is reached                      ("max_turns")
    harness error an unexpected exception anywhere in the loop   ("harness_error")

Messages are plain dicts:
    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": <JSON text of the result>}
A turn without tool calls is answered with a fixed neutral user message and still counts as a turn.

Logging (same JSONL as the tool calls): `episode_start` (the prompts), one `assistant_turn` line per
turn (role "assistant": text, tool calls with their ids; tokens and latency as given by the backend,
null for a fake one; `output.call_ids` links to the tool-call lines of Episode.call, written just
before it), `nudge` for the fixed reply, `episode_end` (stopped_reason) and, on an unexpected
exception, `harness_error`. A run with a harness error is EXCLUDED from statistics and counted
separately (bench.honesty).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bench.tools import Episode

NUDGE = "No tool call was made. Please continue using the available tools."
FIRST_MESSAGE = "Please begin."
PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Any  # a dict normally; whatever the model produced otherwise (Episode.call checks)


@dataclass
class AssistantTurn:
    """One assistant turn returned by a backend."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: dict[str, int] | None = None  # e.g. {"input": 812, "output": 95}; None for a fake
    latency: float | None = None  # seconds; None for a fake


class Backend(Protocol):
    """Messages and tool specs in, one assistant turn out."""

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn: ...


@dataclass
class AgentResult:
    stopped_reason: str  # "submit" | "tool_call_cap" | "max_turns" | "harness_error"
    n_turns: int
    harness_error: str | None = None  # exception type and message, if any
    messages: list[dict[str, Any]] = field(default_factory=list)


def load_system_prompt(name: str = "system_v1") -> str:
    return (PROMPT_DIR / f"{name}.md").read_text().strip()


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": turn.text,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls
        ],
    }


def run_agent(
    episode: Episode,
    backend: Backend,
    system_prompt: str,
    max_turns: int = 60,
    first_message: str = FIRST_MESSAGE,
) -> AgentResult:
    """Play one episode with `backend`; returns how and why it stopped."""
    logger = episode.logger
    tools = episode.tool_specs()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_message},
    ]

    def log(role: str, name: str, arguments: dict, output: Any, **kw: Any) -> None:
        if logger is not None:
            logger.log(role, name, arguments, output, episode.env.budget_remaining, **kw)

    log(
        "harness",
        "episode_start",
        {"max_turns": max_turns, "max_tool_calls": episode.max_tool_calls},
        {
            "system_prompt": system_prompt,
            "first_message": first_message,
            "tools": [t["name"] for t in tools],
        },
    )
    reason, n_turns, error = "max_turns", 0, None
    try:
        for n_turns in range(1, max_turns + 1):
            turn = backend.generate(messages, tools)
            messages.append(_assistant_message(turn))
            call_ids = []
            for call in turn.tool_calls:
                result = episode.call(call.name, call.arguments)
                call_ids.append(episode.last_call_id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result),
                    }
                )
            log(
                "assistant",
                "assistant_turn",
                {
                    "turn": n_turns,
                    "text": turn.text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in turn.tool_calls
                    ],
                },
                {"call_ids": call_ids},
                tokens=turn.tokens,
                latency=turn.latency,
            )
            if episode.submitted:
                reason = "submit"
                break
            if episode.n_tool_calls >= episode.max_tool_calls:
                reason = "tool_call_cap"
                break
            if not turn.tool_calls:
                messages.append({"role": "user", "content": NUDGE})
                log("harness", "nudge", {"turn": n_turns}, {"content": NUDGE})
        else:
            n_turns = max_turns
    except Exception as exc:  # noqa: BLE001 - anything unexpected is a harness error, by design
        reason = "harness_error"
        error = f"{type(exc).__name__}: {exc}"
        log("harness", "harness_error", {"turn": n_turns}, {
            "error": error, "traceback": traceback.format_exc()[-2000:],
        })  # fmt: skip
    log("harness", "episode_end", {}, {"stopped_reason": reason, "n_turns": n_turns})
    return AgentResult(reason, n_turns, error, messages)
