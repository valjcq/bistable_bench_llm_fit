"""Rebuild an agent episode from its run directory alone.

An LLM run is not reproducible, so its log is the record. This module reads `events.jsonl` (and
`config.json`, `result.json` when present) and rebuilds, without any model or environment:

  * the conversation: for every turn, the exact provider-neutral messages the backend received
    (`turn["messages_sent"]`; the system prompt, the first message, the agent's earlier turns, the
    tool messages exactly as the model received them, and the fixed nudges), and
  * what came back and what it caused: reasoning, text, tool calls with their raw arguments,
    finish reason, tokens, latency, attempts, every tool result and the budget after it.

Logs written before the loop recorded `tool_messages` still render, but their tool results are
re-serialised from the tool lines (key order may differ from what the model saw); such an episode
is flagged `exact: False`.

`render_markdown` is what scripts/show_episode.py prints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.logging import read_events


def _raw_arguments(raw: Any, index: int) -> Any:
    """The arguments of the index-th tool call exactly as the provider returned them (a JSON
    string for OpenAI-style APIs, an object for Anthropic), or None if the raw body is absent."""
    try:
        if "choices" in raw:
            return raw["choices"][0]["message"]["tool_calls"][index]["function"]["arguments"]
        uses = [b for b in raw["content"] if b.get("type") == "tool_use"]
        return uses[index]["input"]
    except (KeyError, IndexError, TypeError):
        return None


def rebuild(run_dir: str | Path) -> dict[str, Any]:
    """The episode in `run_dir` as {"setup", "turns", "end", "submit", "exact", ...}."""
    run_dir = Path(run_dir)
    events = read_events(run_dir / "events.jsonl")
    start = next(e for e in events if e["name"] == "episode_start")["output"]
    tool_lines = {
        e["call_id"]: e for e in events if e["call_id"] is not None and e["role"] != "optimizer"
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": start["system_prompt"]},
        {"role": "user", "content": start["first_message"]},
    ]
    turns: list[dict[str, Any]] = []
    exact = True
    for e in events:
        if e["name"] == "assistant_turn":
            args, out = e["arguments"], e["output"]
            sent = [dict(m) for m in messages]
            calls = args["tool_calls"]
            messages.append(
                {
                    "role": "assistant",
                    "content": args["text"],
                    "tool_calls": [dict(c) for c in calls],
                }
            )
            if "tool_messages" in out:
                tool_messages = out["tool_messages"]
            else:  # an older log: rebuilt from the tool lines, key order not guaranteed
                exact = False
                tool_messages = [
                    {
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "name": c["name"],
                        "content": json.dumps(tool_lines[cid]["output"]),
                    }
                    for c, cid in zip(calls, out["call_ids"], strict=True)
                ]
            messages.extend(dict(m) for m in tool_messages)
            raw = out.get("raw_response")
            turns.append(
                {
                    "turn": args["turn"],
                    "messages_sent": sent,
                    "text": args["text"],
                    "reasoning": out.get("reasoning"),
                    "finish_reason": out["meta"].get("finish_reason"),
                    "attempts": out["meta"].get("attempts"),
                    "tokens": e["tokens"],
                    "latency": e["latency"],
                    "raw_response": raw,
                    "calls": [
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "arguments": c["arguments"],
                            "raw_arguments": _raw_arguments(raw, i) if raw else None,
                            "invalid": tool_lines[cid]["invalid"],
                            "budget_after": tool_lines[cid]["budget_after"],
                            "result": m["content"],
                        }
                        for i, (c, cid, m) in enumerate(
                            zip(calls, out["call_ids"], tool_messages, strict=True)
                        )
                    ],
                }
            )
        elif e["name"] == "nudge":
            messages.append({"role": "user", "content": e["output"]["content"]})
    submit = next((e for e in reversed(events) if e["name"] == "submit"), None)
    end = next((e["output"] for e in reversed(events) if e["name"] == "episode_end"), None)
    harness_error = next((e["output"] for e in events if e["name"] == "harness_error"), None)

    def load(name: str) -> Any:
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "setup": start,
        "config": load("config.json"),
        "result": load("result.json"),
        "turns": turns,
        "final_messages": messages,
        "submit": submit,
        "end": end,
        "harness_error": harness_error,
        "exact": exact,
    }


def _fence(text: str, lang: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def render_markdown(run_dir: str | Path) -> str:
    """One episode as Markdown: setup once, then every turn with its results, then the claim."""
    ep = rebuild(run_dir)
    setup, config, result = ep["setup"], ep["config"] or {}, ep["result"] or {}
    template = setup.get("request_template") or {}
    sampling = {k: v for k, v in (template.get("fixed") or {}).items() if k != "tools"}
    out = [f"# Episode {Path(run_dir).name}", ""]
    if result:
        out += [
            f"- task `{result.get('task_id')}`, seed {result.get('seed')}, hidden feasibility: "
            f"**{result.get('feasible')}**, category: **{result.get('category')}**",
            f"- stopped: {result.get('stopped_reason')}, "
            f"evaluations used {result.get('evals_used')}",
        ]
    if not ep["exact"]:
        out.append("- WARNING: older log without `tool_messages`; tool results are re-serialised")
    out += [
        f"- prompt file `{config.get('prompt_file')}` sha256 `{config.get('prompt_sha256')}`",
        f"- sampling as sent: `{json.dumps(sampling)}`",
        f"- endpoint: `{template.get('endpoint')}`",
        f"- server preflight: `{json.dumps(config.get('server_preflight'))}`",
        "",
        "## System prompt",
        _fence(setup["system_prompt"]),
        "",
        f"First user message: {_fence(setup['first_message'])}",
        "",
        "## Tools (as handed to the backend)",
        _fence(json.dumps(setup.get("tool_specs"), indent=1), "json"),
        "",
    ]
    for t in ep["turns"]:
        tok = t["tokens"] or {}
        out += [
            f"## Turn {t['turn']}",
            f"tokens in/out {tok.get('input')}/{tok.get('output')}, latency {t['latency']}, "
            f"attempts {t['attempts']}, finish_reason `{t['finish_reason']}`",
            "",
        ]  # fmt: skip
        if t["reasoning"]:
            out += ["**Reasoning**", _fence(t["reasoning"]), ""]
        if t["text"]:
            out += ["**Text**", _fence(t["text"]), ""]
        for c in t["calls"]:
            raw = c["raw_arguments"]
            out += [
                f"**Call `{c['name']}`** (id `{c['id']}`{', INVALID' if c['invalid'] else ''})",
                _fence(
                    raw if isinstance(raw, str) else json.dumps(c["arguments"], indent=1), "json"
                ),
                "Result as received:",
                _fence(c["result"], "json"),
                f"budget after: {c['budget_after']}",
                "",
            ]  # fmt: skip
    sub = ep["submit"]
    out += ["## End", f"stopped: `{(ep['end'] or {}).get('stopped_reason')}`"]
    if ep["harness_error"]:
        out += ["harness error:", _fence(json.dumps(ep["harness_error"], indent=1), "json")]
    if sub:
        a = sub["arguments"]
        out += [
            f"- claim: **{a.get('outcome')}**",
            f"- params: `{json.dumps(a.get('params'))}`",
            f"- justification: {a.get('justification') or '(none)'}",
            f"- checker verdict: `{json.dumps(sub['output'].get('evaluation_metric'))}`",
        ]  # fmt: skip
    return "\n".join(out) + "\n"
