"""JSONL event log: one line per call, attached to the Environment through its event hook.

Every line has the same fields:

    timestamp, step, role, name, arguments, output, output_sha256, tokens, latency,
    budget_after, seed, config_hash, invalid, call_id, parent_call_id

`call_id` is set on the line a tool call itself writes (Episode.call, or the runner's
run_optimizer); `parent_call_id` is set on every line written WHILE that call runs (the
environment and optimizer events it triggers), so a log can be grouped by call.

`name` is the environment method (evaluate, check, simulate, final_report, budget_exhausted)
or, from checkpoint 7 on, a tool name. `output` is the JSON result; if its JSON text is longer
than `max_output_chars` it is replaced by `output_sha256` (and `output` is null), so the log
stays small but a changed result is still detectable. That limit applies to the internal events
only (environment and optimizer evaluations): lines logged with `keep_full=True` (the tool calls
and assistant turns, i.e. everything the agent sees or says) are never truncated or hashed.
`tokens` and `latency` are null until an
LLM is involved. `budget_after` is the number of evaluations still available after the call.

Two runs are "the same" when their lines are equal after dropping the fields in VOLATILE_FIELDS
(timestamp, latency): everything else, including step order, must reproduce exactly.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

VOLATILE_FIELDS = ("timestamp", "latency")

# Keys of an Environment event that are the call's inputs; the remaining keys are its output.
_ARGUMENT_KEYS = ("params", "objective", "x0", "T", "attempted")
_BOOKKEEPING_KEYS = ("event", "budget_remaining")


def _json_default(o: Any) -> Any:
    """Make numpy scalars/arrays JSON-serializable."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


class JsonlLogger:
    """Writes one JSON line per event and flushes it at once (a crashed run keeps its log).

    Pass `logger.env_hook` as the Environment's `event_hook`. Other code (the tool layer)
    calls `logger.log(...)` directly for events that are not environment calls.
    """

    def __init__(
        self,
        path: str | Path,
        seed: int,
        config_hash: str,
        role: str = "baseline",
        max_output_chars: int = 20_000,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w")
        self.seed = seed
        self.config_hash = config_hash
        self.role = role  # who issues the environment calls (the harness reports itself)
        self.max_output_chars = max_output_chars
        self.step = 0
        self._next_call_id = 1
        self.parent_call_id: int | None = None  # call in progress, if any

    def begin_call(self) -> int:
        """Start a tool call: lines logged until `end_call` get this id as parent_call_id."""
        call_id = self._next_call_id
        self._next_call_id += 1
        self.parent_call_id = call_id
        return call_id

    def end_call(self) -> None:
        self.parent_call_id = None

    def log(
        self,
        role: str,
        name: str,
        arguments: dict[str, Any],
        output: Any,
        budget_after: int | None,
        invalid: bool = False,
        call_id: int | None = None,
        tokens: Any = None,  # an int, or e.g. {"input": ..., "output": ...} from an LLM backend
        latency: float | None = None,
        keep_full: bool = False,  # agent-visible content: no size limit, never hashed
    ) -> None:
        text = _canonical(output)
        too_big = not keep_full and len(text) > self.max_output_chars
        line = {
            "timestamp": time.time(),
            "step": self.step,
            "role": role,
            "name": name,
            "arguments": arguments,
            "output": None if too_big else json.loads(text),
            "output_sha256": hashlib.sha256(text.encode()).hexdigest() if too_big else None,
            "tokens": tokens,
            "latency": latency,
            "budget_after": budget_after,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "invalid": invalid,
            "call_id": call_id,
            "parent_call_id": self.parent_call_id,
        }
        self.step += 1
        self._file.write(json.dumps(line, default=_json_default) + "\n")
        self._file.flush()

    def env_hook(self, event: dict[str, Any]) -> None:
        """Environment event_hook: turns an Environment event into one log line."""
        arguments = {k: event[k] for k in _ARGUMENT_KEYS if k in event}
        output = {k: v for k, v in event.items() if k not in _ARGUMENT_KEYS + _BOOKKEEPING_KEYS}
        role = "harness" if event["event"] == "final_report" else self.role
        self.log(role, event["event"], arguments, output, event["budget_remaining"])

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """All lines of a JSONL log as dicts."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def comparable(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events without the volatile fields: two reproducible runs give equal results here."""
    return [{k: v for k, v in e.items() if k not in VOLATILE_FIELDS} for e in events]
