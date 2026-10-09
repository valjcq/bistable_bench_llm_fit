"""LLM backends for the agent loop: OpenAI-compatible chat completions and the Anthropic Messages
API, both with native tool calling.

A backend turns the loop's provider-neutral messages and our JSON-Schema tool specs into one HTTP
request, and the response into an `AssistantTurn` (text, tool calls, tokens, latency). Only the
standard library is used (urllib); the HTTP layer is a small `Transport` interface so tests can
mock it, and no test ever touches the network.

- `OpenAICompatibleBackend`: POST {base_url}/chat/completions. Works with vLLM, llama.cpp's
  llama-server and hosted OpenAI-style endpoints. The API key, if any, comes from the environment
  variable NAMED in the config (`api_key_env`); a local server without a key needs none.
- `AnthropicBackend`: POST {base_url}/v1/messages with `x-api-key`.

Retries: rate limits (429), server errors (5xx, including Anthropic's 529) and connection errors
are retried with exponential backoff (2 s, 4 s, ... up to 60 s; a numeric Retry-After header is
honoured up to 120 s) up to `max_retries` attempts in total. After that, or at once for any other
error status (bad request, authentication), the backend raises `BackendError`, which the agent
loop records as a `harness_error` (the run is excluded from the statistics).

Secrets: the key is read from the environment when the backend is built and only ever placed in
the request header. It is not stored on the backend object's repr, not logged, not in any config;
error messages quote the response body, never the request headers.

Reproducibility: LLM runs are not bit-reproducible (see docs/agent.md); the logged transcript is
the record.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from bench.agent.loop import AssistantTurn, ToolCall
from bench.config import AgentConfig

ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
BASE_DELAY, MAX_DELAY, MAX_RETRY_AFTER = 2.0, 60.0, 120.0


class BackendError(RuntimeError):
    """A request failed for good (retries exhausted, or a non-retryable error)."""


class TransportError(RuntimeError):
    """Connection problem or timeout (retryable)."""


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)  # lower-case names
    body: Any = None  # parsed JSON if possible, else the text


class Transport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpResponse: ...

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse: ...


class UrllibTransport:
    """The real HTTP transport (standard library)."""

    def post(
        self, url: str, headers: Mapping[str, str], payload: dict[str, Any], timeout: float
    ) -> HttpResponse:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=dict(headers), method="POST"
        )
        return self._send(req, timeout)

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        return self._send(urllib.request.Request(url, headers=dict(headers), method="GET"), timeout)

    def _send(self, req: urllib.request.Request, timeout: float) -> HttpResponse:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return _response(resp.status, resp.headers, resp.read())
        except urllib.error.HTTPError as e:  # a status code: not an exception for us
            return _response(e.code, e.headers, e.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise TransportError(f"{type(e).__name__}: {e}") from e


def _response(status: int, headers: Any, raw: bytes) -> HttpResponse:
    text = raw.decode("utf-8", errors="replace")
    try:
        body: Any = json.loads(text)
    except ValueError:
        body = text
    return HttpResponse(status, {k.lower(): v for k, v in headers.items()}, body)


# ---- retries ---------------------------------------------------------------------------------


def post_with_retries(
    transport: Transport,
    url: str,
    headers: Mapping[str, str],
    payload: dict[str, Any],
    timeout: float,
    max_attempts: int,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> tuple[HttpResponse, int]:
    """POST with exponential backoff; returns (response, attempts used). Raises BackendError."""
    return request_with_retries(
        lambda: transport.post(url, headers, payload, timeout), max_attempts, sleep, jitter
    )


def request_with_retries(
    send: Callable[[], HttpResponse],
    max_attempts: int,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> tuple[HttpResponse, int]:
    """Call `send()` until it returns status 200, retrying 429, 5xx and transport errors with
    exponential backoff (see the module docstring). Returns (response, attempts used)."""
    last = "no attempt"
    for attempt in range(1, max_attempts + 1):
        retry_after = None
        try:
            resp = send()
        except TransportError as e:
            last = str(e)
        else:
            if resp.status == 200:
                return resp, attempt
            last = f"HTTP {resp.status}: {_excerpt(resp.body)}"
            if resp.status != 429 and resp.status < 500:
                raise BackendError(f"non-retryable error, {last}")
            retry_after = resp.headers.get("retry-after")
        if attempt < max_attempts:
            delay = min(MAX_DELAY, BASE_DELAY * 2 ** (attempt - 1))
            try:
                delay = max(delay, min(MAX_RETRY_AFTER, float(retry_after)))
            except (TypeError, ValueError):
                pass
            sleep(delay * (1 + 0.1 * jitter()))
    raise BackendError(f"{max_attempts} attempts failed, last: {last}")


def _excerpt(body: Any, n: int = 300) -> str:
    return (body if isinstance(body, str) else json.dumps(body))[:n]


# ---- conversion of our schemas and messages ----------------------------------------------------


def _strip_schema(schema: Any) -> Any:
    """Our tool schemas declare "$schema"; providers do not need (some reject) it."""
    if isinstance(schema, dict):
        return {k: _strip_schema(v) for k, v in schema.items() if k != "$schema"}
    if isinstance(schema, list):
        return [_strip_schema(v) for v in schema]
    return schema


def to_openai_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": _strip_schema(s["input_schema"]),
            },
        }
        for s in specs
    ]


def to_anthropic_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": _strip_schema(s["input_schema"]),
        }
        for s in specs
    ]


def _arguments_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m["content"] or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": _arguments_text(c["arguments"]),
                        },
                    }
                    for c in m["tool_calls"]
                ]
            elif not m["content"]:
                msg["content"] = ""
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """(system prompt, messages). Tool results become tool_result blocks of a user message;
    consecutive ones are merged so the roles alternate."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks: list[dict[str, Any]] = []
            if m["content"]:
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls", []):
                args = c["arguments"] if isinstance(c["arguments"], dict) else {}
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": args})
            out.append(
                {
                    "role": "assistant",
                    "content": blocks or [{"type": "text", "text": "(no output)"}],
                }
            )
        else:  # tool result
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


def parse_arguments(raw: Any) -> Any:
    """Tool-call arguments as the model produced them: a dict if they parse to one, otherwise the
    raw text (Episode.call then refuses it as an invalid call, which is recorded)."""
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    return value if isinstance(value, dict) else raw


# ---- the backends ------------------------------------------------------------------------------


class _Backend:
    def __init__(
        self,
        cfg: AgentConfig,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        environ: Mapping[str, str] | None = None,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._cfg = cfg
        self._transport = transport or UrllibTransport()
        self._sleep, self._jitter = sleep, jitter
        env = os.environ if environ is None else environ
        self._key = None
        if cfg.api_key_env:
            self._key = env.get(cfg.api_key_env)
            if not self._key:
                raise BackendError(f"environment variable {cfg.api_key_env} is not set")

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self._cfg.model!r})"

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> tuple[Any, int]:
        resp, attempts = post_with_retries(
            self._transport, url, headers, payload, self._cfg.timeout_s, self._cfg.max_retries,
            self._sleep, self._jitter,
        )  # fmt: skip
        return resp.body, attempts


class OpenAICompatibleBackend(_Backend):
    """Chat completions with native tool calling (vLLM, llama.cpp, hosted OpenAI-style APIs)."""

    def request_template(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """What every request of the episode has in common, exactly as sent: the endpoint and the
        `fixed` payload fields (model, tools in the provider's format, the sampling parameters
        actually sent). generate() builds its payloads from this, so the record cannot drift from
        what goes over the wire; only `messages` changes between requests."""
        fixed: dict[str, Any] = {
            "model": self._cfg.model,
            "tools": to_openai_tools(tools),
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        for name in ("top_p", "top_k", "min_p"):  # optional: sent only when set
            if getattr(self._cfg, name) is not None:
                fixed[name] = getattr(self._cfg, name)
        return {"endpoint": f"{(self._cfg.base_url or '').rstrip('/')}/chat/completions",
                "fixed": fixed}  # fmt: skip

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        template = self.request_template(tools)
        payload = {**template["fixed"], "messages": to_openai_messages(messages)}
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        url = template["endpoint"]
        start = time.perf_counter()
        body, attempts = self._post(url, headers, payload)
        latency = time.perf_counter() - start
        try:
            choice = body["choices"][0]
            msg = choice["message"]
            calls = [
                ToolCall(
                    tc.get("id") or f"call_{i}",
                    tc["function"]["name"],
                    parse_arguments(tc["function"].get("arguments")),
                )
                for i, tc in enumerate(msg.get("tool_calls") or [], start=1)
            ]
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"unexpected response shape ({e!r}): {_excerpt(body)}") from e
        usage = body.get("usage") or {}
        return AssistantTurn(
            text=msg.get("content") or "",
            tool_calls=calls,
            tokens={"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")},
            latency=latency,
            reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
            meta={"finish_reason": choice.get("finish_reason"), "attempts": attempts},
            raw=body,
        )


class AnthropicBackend(_Backend):
    """The Messages API with native tool use."""

    def request_template(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """The fields every request of the episode shares, exactly as sent (see the OpenAI one)."""
        fixed: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "tools": to_anthropic_tools(tools),
        }
        for name in ("top_p", "top_k"):  # optional: sent only when set (min_p does not exist here)
            if getattr(self._cfg, name) is not None:
                fixed[name] = getattr(self._cfg, name)
        return {"endpoint": f"{(self._cfg.base_url or ANTHROPIC_URL).rstrip('/')}/v1/messages",
                "fixed": fixed}  # fmt: skip

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantTurn:
        system, conv = to_anthropic_messages(messages)
        template = self.request_template(tools)
        payload = {**template["fixed"], "system": system, "messages": conv}
        headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if self._key:
            headers["x-api-key"] = self._key
        url = template["endpoint"]
        start = time.perf_counter()
        body, attempts = self._post(url, headers, payload)
        latency = time.perf_counter() - start
        try:
            blocks = body["content"]
            text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
            thinking = "\n".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking")
            calls = [
                ToolCall(b["id"], b["name"], b.get("input", {}))
                for b in blocks
                if b["type"] == "tool_use"
            ]
        except (KeyError, TypeError) as e:
            raise BackendError(f"unexpected response shape ({e!r}): {_excerpt(body)}") from e
        usage = body.get("usage") or {}
        return AssistantTurn(
            text=text,
            tool_calls=calls,
            tokens={"input": usage.get("input_tokens"), "output": usage.get("output_tokens")},
            latency=latency,
            reasoning=thinking or None,
            meta={"finish_reason": body.get("stop_reason"), "attempts": attempts},
            raw=body,
        )


def preflight(
    cfg: AgentConfig,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ask an OpenAI-compatible server what it is serving, and refuse a mismatch.

    Run ONLY by the real runner (never in --dry-run, never in tests against a real server). It
    queries GET {base_url}/models and, for llama.cpp, GET <server root>/props (a failure of /props
    is ignored: vLLM and hosted APIs do not have it). The reported names are every model id, the
    file name of each id and the file name of `model_path` in /props. If `agent.expected_model` is
    not among them, BackendError is raised with a clear message, before any episode. Returns what
    was found (to be recorded in each run's config.json): the model ids, the llama.cpp props that
    matter (model_path, alias, context per slot, number of slots) and the match.
    """
    if cfg.backend != "openai_compatible":
        return {}
    if not cfg.expected_model:
        raise BackendError("agent.expected_model is required for a real openai_compatible run")
    transport = transport or UrllibTransport()
    env = os.environ if environ is None else environ
    headers: dict[str, str] = {}
    if cfg.api_key_env:
        if not env.get(cfg.api_key_env):
            raise BackendError(f"environment variable {cfg.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {env[cfg.api_key_env]}"
    base = (cfg.base_url or "").rstrip("/")
    resp, _ = request_with_retries(
        lambda: transport.get(f"{base}/models", headers, cfg.timeout_s), cfg.max_retries, sleep
    )
    try:
        ids = [m["id"] for m in resp.body["data"]]
    except (KeyError, TypeError) as e:
        raise BackendError(f"preflight: unexpected /models response: {_excerpt(resp.body)}") from e
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    props: dict[str, Any] = {}
    try:
        p = transport.get(f"{root}/props", headers, cfg.timeout_s)
        if p.status == 200 and isinstance(p.body, dict):
            gen = p.body.get("default_generation_settings") or {}
            props = {
                "model_path": p.body.get("model_path"),
                "model_alias": p.body.get("model_alias"),
                "n_ctx_per_slot": gen.get("n_ctx"),
                "total_slots": p.body.get("total_slots"),
            }
    except TransportError:
        pass
    reported = set(ids) | {os.path.basename(i) for i in ids}
    if props.get("model_path"):
        reported |= {props["model_path"], os.path.basename(props["model_path"])}
    if props.get("model_alias"):
        reported.add(props["model_alias"])
    if cfg.expected_model not in reported:
        raise BackendError(
            f"preflight: the server at {base} reports {sorted(reported)} but agent.expected_model "
            f"is {cfg.expected_model!r}; nothing was run"
        )
    return {
        "models_endpoint": ids,
        "props": props,
        "expected_model": cfg.expected_model,
        "matched": True,
        "queried_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def make_backend(cfg: AgentConfig, **kwargs: Any) -> Any:
    """The backend described by an `agent` config section (kwargs: transport, sleep, environ)."""
    if cfg.backend == "openai_compatible":
        return OpenAICompatibleBackend(cfg, **kwargs)
    if cfg.backend == "anthropic":
        return AnthropicBackend(cfg, **kwargs)
    if cfg.backend == "fake":
        from bench.agent.fake import reference_backend

        return reference_backend()
    raise ValueError(f"unknown backend {cfg.backend!r}")
