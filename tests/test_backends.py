"""LLM backends, the agent config, and the agent runner. Mocked HTTP only: no test touches the
network or any server, and no test reads the example config for a real server."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from bench.agent.backends import (
    AnthropicBackend,
    BackendError,
    HttpResponse,
    OpenAICompatibleBackend,
    TransportError,
    make_backend,
    parse_arguments,
    post_with_retries,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)
from bench.agent.loop import run_agent
from bench.agent.run import main as run_main
from bench.agent.run import resolve_agent, run_one, usage_summary
from bench.config import AgentConfig, BudgetConfig, config_from_dict, load_config
from bench.honesty import summarize
from bench.tasks import load_task

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-test-SECRET-value-0123456789"
TOY = ROOT / "tasks/honesty/toy_1pop_K2_seed302.json"


class MockTransport:
    """Returns queued responses (or raises queued exceptions) and records every request."""

    def __init__(self, *queue, gets=None):
        self.queue, self.requests = list(queue), []
        self.gets, self.get_requests = dict(gets or {}), []  # GET url -> response / exception

    def post(self, url, headers, payload, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "payload": payload})
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _get(self, url, headers, timeout):
    self.get_requests.append({"url": url, "headers": dict(headers)})
    item = self.gets.get(url, HttpResponse(404, {}, "not found"))
    if isinstance(item, Exception):
        raise item
    return item


MockTransport.get = _get


class NoNetwork:
    """A transport that fails the test if it is used at all."""

    def post(self, *a, **k):
        raise AssertionError("the network was used")

    get = post


def ok(body):
    return HttpResponse(200, {}, body)


def openai_reply(text="", calls=(), usage=(11, 7), reasoning=None):
    msg = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [
            {"id": f"id{i}", "type": "function", "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(calls, start=1)
        ]
    if reasoning:
        msg["reasoning_content"] = reasoning
    return ok({
        "choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    })  # fmt: skip


def anthropic_reply(blocks, usage=(21, 5)):
    return ok({
        "content": blocks, "stop_reason": "tool_use",
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    })  # fmt: skip


def agent_cfg(backend="openai_compatible", **kw):
    base = dict(backend=backend, model="m-1", base_url="http://host:8001/v1")
    if backend == "anthropic":
        base = dict(backend=backend, model="claude-x", api_key_env="ANTHROPIC_API_KEY")
    return AgentConfig(**{**base, **kw})


MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "name": "get_task", "arguments": {}},
        {"id": "b", "name": "get_budget", "arguments": "not json"},
    ]},
    {"role": "tool", "tool_call_id": "a", "name": "get_task", "content": '{"K": 2}'},
    {"role": "tool", "tool_call_id": "b", "name": "get_budget", "content": '{"error": 1}'},
    {"role": "assistant", "content": "thinking", "tool_calls": []},
    {"role": "user", "content": "nudge"},
]  # fmt: skip
SPECS = [{
    "name": "t", "description": "d",
    "input_schema": {
        "$schema": "x", "type": "object", "properties": {"p": {"type": ["object", "null"]}},
    },
}]  # fmt: skip


# ---- conversion --------------------------------------------------------------------------------


def test_tool_conversion_strips_the_schema_keyword_and_keeps_the_rest():
    o = to_openai_tools(SPECS)[0]
    assert o["type"] == "function" and o["function"]["name"] == "t"
    assert "$schema" not in o["function"]["parameters"]
    assert o["function"]["parameters"]["properties"]["p"]["type"] == ["object", "null"]
    a = to_anthropic_tools(SPECS)[0]
    assert set(a) == {"name", "description", "input_schema"} and "$schema" not in a["input_schema"]
    assert "$schema" in SPECS[0]["input_schema"]  # the original is untouched


def test_openai_messages():
    out = to_openai_messages(MESSAGES)
    assert [m["role"] for m in out] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
    ]
    calls = out[2]["tool_calls"]
    assert out[2]["content"] is None and calls[0]["function"]["arguments"] == "{}"
    assert calls[1]["function"]["arguments"] == "not json" and calls[0]["type"] == "function"
    assert out[3] == {"role": "tool", "tool_call_id": "a", "content": '{"K": 2}'}
    assert out[5] == {"role": "assistant", "content": "thinking"}


def test_anthropic_messages_merge_tool_results_and_extract_the_system_prompt():
    system, out = to_anthropic_messages(MESSAGES)
    assert system == "SYS"
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]
    blocks = out[1]["content"]
    assert [b["type"] for b in blocks] == ["tool_use", "tool_use"]
    assert blocks[0]["input"] == {} and blocks[1]["input"] == {}  # unparsable arguments -> {}
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["a", "b"]  # merged into one user message
    assert out[3]["content"] == [{"type": "text", "text": "thinking"}]
    empty = to_anthropic_messages([{"role": "assistant", "content": "", "tool_calls": []}])[1]
    assert empty[0]["content"][0]["text"] == "(no output)"


def test_parse_arguments():
    assert parse_arguments('{"a": 1}') == {"a": 1} and parse_arguments("") == {}
    assert parse_arguments({"a": 1}) == {"a": 1} and parse_arguments(None) == {}
    assert parse_arguments("not json") == "not json" and parse_arguments("[1]") == "[1]"


# ---- the OpenAI-compatible backend -------------------------------------------------------------


def test_openai_request_and_response():
    t = MockTransport(openai_reply("hi", [("get_task", "{}"), ("submit", "{oops")],
                                   reasoning="because"))  # fmt: skip
    cfg = agent_cfg(api_key_env="MY_KEY", temperature=0.2, max_tokens=99)
    b = OpenAICompatibleBackend(cfg, transport=t, environ={"MY_KEY": SECRET})
    turn = b.generate(MESSAGES, SPECS)
    req = t.requests[0]
    assert req["url"] == "http://host:8001/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {SECRET}"
    p = req["payload"]
    assert (p["model"], p["temperature"], p["max_tokens"]) == ("m-1", 0.2, 99)
    assert p["tools"][0]["function"]["name"] == "t" and p["messages"][0]["role"] == "system"
    assert turn.text == "hi" and turn.reasoning == "because"
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("id1", "get_task", {}), ("id2", "submit", "{oops"),
    ]  # fmt: skip
    assert (
        turn.tokens == {"input": 11, "output": 7} and turn.latency is not None and turn.latency >= 0
    )
    assert turn.meta == {"finish_reason": "tool_calls", "attempts": 1}
    assert SECRET not in repr(b)


def test_openai_without_a_key_sends_no_authorization_header():
    t = MockTransport(openai_reply("x"))
    OpenAICompatibleBackend(agent_cfg(base_url="http://h/v1/"), transport=t).generate(
        MESSAGES, SPECS
    )
    assert "Authorization" not in t.requests[0]["headers"]
    assert t.requests[0]["url"] == "http://h/v1/chat/completions"


def test_missing_key_variable_is_an_error_not_a_silent_unauthenticated_call():
    with pytest.raises(BackendError, match="MY_KEY"):
        OpenAICompatibleBackend(
            agent_cfg(api_key_env="MY_KEY"), transport=MockTransport(), environ={}
        )


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"nope": 1}]}, "<html>"])
def test_malformed_openai_responses_raise_backend_error(body):
    b = OpenAICompatibleBackend(agent_cfg(), transport=MockTransport(ok(body)))
    with pytest.raises(BackendError, match="unexpected response"):
        b.generate(MESSAGES, SPECS)


# ---- the Anthropic backend ---------------------------------------------------------------------


def test_anthropic_request_and_response():
    t = MockTransport(anthropic_reply([
        {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "tu1", "name": "get_task", "input": {}},
    ]))  # fmt: skip
    b = AnthropicBackend(agent_cfg("anthropic"), transport=t, environ={"ANTHROPIC_API_KEY": SECRET})
    turn = b.generate(MESSAGES, SPECS)
    req = t.requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert (
        req["headers"]["x-api-key"] == SECRET
        and req["headers"]["anthropic-version"] == "2023-06-01"
    )
    p = req["payload"]
    assert p["system"] == "SYS" and p["temperature"] == 0.0 and p["max_tokens"] == 4096
    assert p["tools"][0]["name"] == "t" and p["messages"][0]["role"] == "user"
    assert turn.text == "ok" and turn.reasoning == "hmm"
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [("tu1", "get_task", {})]
    assert turn.tokens == {"input": 21, "output": 5} and turn.meta["finish_reason"] == "tool_use"


def test_make_backend_dispatch():
    assert isinstance(make_backend(agent_cfg(), transport=MockTransport()), OpenAICompatibleBackend)
    b = make_backend(
        agent_cfg("anthropic"), transport=MockTransport(), environ={"ANTHROPIC_API_KEY": "k"}
    )
    assert isinstance(b, AnthropicBackend)
    fake = make_backend(AgentConfig("fake", "fake"))
    assert (
        fake.generate([], []).tool_calls == [] or True
    )  # a FakeBackend; its script is tested elsewhere


# ---- retries -----------------------------------------------------------------------------------


def run_post(transport, attempts=5):
    sleeps = []
    out = post_with_retries(transport, "u", {}, {}, 1.0, attempts, sleeps.append, lambda: 0.0)
    return out, sleeps


def test_retries_on_rate_limit_and_server_errors_with_backoff_and_retry_after():
    t = MockTransport(
        HttpResponse(429, {"retry-after": "7"}, "slow down"),
        HttpResponse(500, {}, "oops"),
        TransportError("timeout"),
        ok({"fine": True}),
    )
    (resp, attempts), sleeps = run_post(t)
    assert resp.body == {"fine": True} and attempts == 4
    assert sleeps == [7.0, 4.0, 8.0]  # max(2, Retry-After 7); 2*2; 2*4


def test_non_retryable_error_fails_at_once():
    t = MockTransport(HttpResponse(400, {}, {"error": "bad request"}), ok({}))
    with pytest.raises(BackendError, match="non-retryable.*400"):
        run_post(t)
    assert len(t.requests) == 1


def test_n_failures_raise_backend_error_and_the_loop_records_a_harness_error(tmp_path):
    t = MockTransport(*[HttpResponse(503, {}, "down")] * 3)
    cfg = AgentConfig("openai_compatible", "m", base_url="http://h/v1", max_retries=3)
    sleeps = []
    backend = OpenAICompatibleBackend(cfg, transport=t, sleep=sleeps.append, jitter=lambda: 0.0)
    with pytest.raises(BackendError, match="3 attempts failed"):
        backend.generate(MESSAGES, SPECS)
    assert sleeps == [2.0, 4.0]
    # through the whole pipeline: a harness error, excluded from the statistics
    t2 = MockTransport(*[HttpResponse(503, {}, "down")] * 3)
    exp = replace(
        load_config(ROOT / "configs/exp_toy_1pop.yaml"), agent=cfg, budget=BudgetConfig(60)
    )
    task = load_task(TOY)
    from bench.honesty import run_honesty_episode

    def play(ep):
        b = OpenAICompatibleBackend(cfg, transport=t2, sleep=lambda s: None)
        return run_agent(ep, b, "SYS")

    row = run_honesty_episode(exp, task, "t", 0, tmp_path, "x", 40, play)
    assert row["harness_error"] is True and row["stopped_reason"] == "harness_error"
    line = [ln for ln in summarize([row]).splitlines() if ln.startswith("| feasible")][0]
    assert [c.strip() for c in line.strip("|").split("|")][1] == "0"  # no counted run


# ---- an episode through a mocked HTTP backend: usage logged, key never written -----------------


def test_full_episode_through_mocked_http_logs_usage_and_never_the_key(tmp_path):
    exp = replace(load_config(ROOT / "configs/exp_toy_1pop.yaml"), budget=BudgetConfig(60))
    cfg = agent_cfg(api_key_env="MY_KEY", max_tool_calls=10)
    exp = replace(exp, agent=cfg)
    t = MockTransport(
        openai_reply("", [("get_task", "{}")], usage=(100, 10)),
        openai_reply(
            "",
            [("submit", json.dumps({"outcome": "no_solution", "justification": "j"}))],
            usage=(150, 12),
        ),  # fmt: skip
    )
    task = load_task(TOY)
    from bench.honesty import run_honesty_episode

    def play(ep):
        b = OpenAICompatibleBackend(cfg, transport=t, environ={"MY_KEY": SECRET})
        return run_agent(ep, b, "SYS", max_turns=cfg.max_turns)

    row = run_honesty_episode(exp, task, "t", 0, tmp_path, "m-1", 10, play)
    assert row["category"] == "no_solution_miss" and row["stopped_reason"] == "submit"
    assert (row["tokens_in"], row["tokens_out"], row["n_turns"], row["retries"]) == (250, 22, 2, 0)
    assert row["model_latency_s"] >= 0
    run_dir = tmp_path / "t__m-1__s0"
    for f in run_dir.iterdir():  # the key value is nowhere in the run directory
        assert SECRET not in f.read_text()
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["experiment"]["agent"]["api_key_env"] == "MY_KEY"  # the NAME is recorded
    assert saved["experiment"]["agent"]["model"] == "m-1"
    assert SECRET not in json.dumps(t.requests[0]["payload"])  # and not in the request body either
    turns = [json.loads(ln) for ln in (run_dir / "events.jsonl").read_text().splitlines()]
    turns = [e for e in turns if e["name"] == "assistant_turn"]
    assert [e["tokens"] for e in turns] == [
        {"input": 100, "output": 10},
        {"input": 150, "output": 12},
    ]
    assert all(isinstance(e["latency"], float) for e in turns)


# ---- the agent config --------------------------------------------------------------------------


def test_agent_config_validation():
    with pytest.raises(ValueError, match="backend"):
        AgentConfig("nope", "m")
    with pytest.raises(ValueError, match="base_url"):
        AgentConfig("openai_compatible", "m")
    with pytest.raises(ValueError, match="NAME"):
        AgentConfig("anthropic", "m", api_key_env="sk-ant-api03-abcdef")  # a pasted key
    with pytest.raises(ValueError, match="NAME"):
        AgentConfig("anthropic", "m", api_key_env="MixedCaseKey123")
    with pytest.raises(ValueError):
        AgentConfig("anthropic", "m", max_turns=0)
    AgentConfig("anthropic", "m", api_key_env="ANTHROPIC_API_KEY")


def test_agent_section_is_in_the_hash_and_absent_when_not_used(cfg_toy):
    assert "agent" not in cfg_toy.resolved()  # existing configs and their hashes are unchanged
    a = replace(cfg_toy, agent=agent_cfg())
    assert a.resolved()["agent"]["model"] == "m-1" and a.hash() != cfg_toy.hash()
    b = replace(cfg_toy, agent=agent_cfg(model="m-2"))
    assert a.hash() != b.hash()  # every field counts
    again = config_from_dict(a.resolved(), a.model_spec, resolved=True)
    assert again.agent == a.agent and again.hash() == a.hash()
    assert a.generation_hash() == cfg_toy.generation_hash()  # tasks stay valid


def test_agent_section_loads_from_yaml(tmp_path):
    src = (ROOT / "configs/exp_toy_1pop.yaml").read_text()
    (tmp_path / "toy_1pop.yaml").write_text((ROOT / "configs/toy_1pop.yaml").read_text())
    (tmp_path / "exp.yaml").write_text(
        src
        + "\nagent:\n  backend: anthropic\n  model: claude-x\n  api_key_env: ANTHROPIC_API_KEY\n"
    )
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg.agent.backend == "anthropic" and cfg.agent.temperature == 0.0
    (tmp_path / "bad.yaml").write_text(
        src + "\nagent:\n  backend: anthropic\n  model: m\n  typo: 1\n"
    )
    with pytest.raises(ValueError, match="unknown key"):
        load_config(tmp_path / "bad.yaml")


# ---- the runner (dry-run only) -----------------------------------------------------------------


def test_dry_run_single_episode_and_batch(tmp_path, capsys):
    out = tmp_path / "runs"
    base = [
        str(ROOT / "configs/exp_toy_1pop.yaml"),
        "--dry-run",
        "--budget",
        "60",
        "--out",
        str(out),
    ]
    run_main([*base, "--task", str(TOY), "--seed", "0"])
    d = out / "toy_1pop_K2_seed302__dryrun__s0"
    assert {p.name for p in d.iterdir()} == {"events.jsonl", "config.json", "result.json"}
    saved = json.loads((d / "config.json").read_text())
    assert saved["experiment"]["agent"]["backend"] == "fake" and len(saved["prompt_sha256"]) == 64
    assert saved["prompt_file"] == "prompts/system_v1.md"
    run_main(
        [
            *base,
            "--tasks",
            str(ROOT / "tasks/honesty/toy_1pop_K2_seed30[01].json"),
            "--seeds",
            "0-1",
        ]
    )
    text = capsys.readouterr().out
    assert "correct_no_solution" in text and "episodes" in text and "config hash" in text
    assert (out / "results.csv").exists() and (out / "results_usage.md").exists()
    assert len(list(out.glob("*__dryrun__s*"))) == 5  # 1 + 2 tasks x 2 seeds


def test_runner_needs_an_agent_section_unless_dry_run(cfg_toy, tmp_path):
    with pytest.raises(SystemExit):
        resolve_agent(cfg_toy, None, dry_run=False)
    f = tmp_path / "agent.yaml"
    f.write_text(yaml.safe_dump({"agent": {"backend": "anthropic", "model": "claude-x"}}))
    assert resolve_agent(cfg_toy, str(f), dry_run=False).agent.model == "claude-x"
    f.write_text(yaml.safe_dump({"backend": "anthropic", "model": "claude-y"}))  # the bare section
    assert resolve_agent(cfg_toy, str(f), dry_run=False).agent.model == "claude-y"
    dry = resolve_agent(cfg_toy, str(f), dry_run=True).agent
    assert dry.backend == "fake" and dry.api_key_env is None


def test_run_one_with_the_dry_run_backend_is_a_correct_episode(cfg_toy, tmp_path):
    cfg = resolve_agent(replace(cfg_toy, budget=BudgetConfig(80)), None, dry_run=True)
    row = run_one(cfg, TOY, 0, tmp_path)
    assert row["category"] in ("correct_solution", "no_solution_miss") and row["n_turns"] == 4


def test_usage_summary():
    rows = [
        {"harness_error": False, "n_turns": 4, "tokens_in": 100, "tokens_out": 10,
         "model_latency_s": 2.0, "retries": 1, "wall_clock_s": 3.0},
        {"harness_error": True, "n_turns": 2, "tokens_in": 50, "tokens_out": 5,
         "model_latency_s": 1.0, "retries": 0, "wall_clock_s": 1.0},
    ]  # fmt: skip
    line = usage_summary(rows).splitlines()[-1]
    assert line.startswith("| 2 | 1 | 6 / 3.0 | 150 / 75 | 15 / 8 | 3.0 | 1 | 4.0 / 2.0 |")
    assert usage_summary([]) == "no episodes\n"


# ---- nothing depends on the example server -----------------------------------------------------


def test_example_config_is_fully_commented_out_and_nothing_refers_to_the_server():
    lines = (ROOT / "configs/agent_llamacpp_example.yaml").read_text().splitlines()
    assert all(ln.strip() == "" or ln.lstrip().startswith("#") for ln in lines)
    assert yaml.safe_load("\n".join(lines)) is None
    # no source file mentions the server's port: nothing in the repo depends on it
    assert "8001" not in "".join(p.read_text() for p in (ROOT / "src").rglob("*.py"))


# ---- sampling parameters -----------------------------------------------------------------------


def test_sampling_parameters_are_validated_and_sent_only_when_set():
    for bad in ({"top_p": 0.0}, {"top_p": 1.5}, {"top_k": -1}, {"min_p": 1.5}):
        with pytest.raises(ValueError):
            agent_cfg(**bad)
    with pytest.raises(ValueError, match="min_p"):
        agent_cfg("anthropic", min_p=0.0)
    plain = MockTransport(openai_reply("x"))
    OpenAICompatibleBackend(agent_cfg(), transport=plain).generate(MESSAGES, SPECS)
    assert not {"top_p", "top_k", "min_p"} & set(plain.requests[0]["payload"])
    qwen = MockTransport(openai_reply("x"))
    cfg = agent_cfg(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
    OpenAICompatibleBackend(cfg, transport=qwen).generate(MESSAGES, SPECS)
    p = qwen.requests[0]["payload"]
    assert (p["temperature"], p["top_p"], p["top_k"], p["min_p"]) == (
        0.6,
        0.95,
        20,
        0.0,
    )  # 0.0 sent
    ant = MockTransport(anthropic_reply([{"type": "text", "text": "x"}]))
    b = AnthropicBackend(agent_cfg("anthropic", top_p=0.9, top_k=5), transport=ant,
                         environ={"ANTHROPIC_API_KEY": "k"})  # fmt: skip
    b.generate(MESSAGES, SPECS)
    assert (ant.requests[0]["payload"]["top_p"], ant.requests[0]["payload"]["top_k"]) == (0.9, 5)
    assert "min_p" not in ant.requests[0]["payload"]


# ---- the preflight (mocked HTTP) ---------------------------------------------------------------

GGUF = "/opt/shared/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
BASE = "http://host:8001/v1"


def llama_server(model_path=GGUF):
    """What llama-server answers: /v1/models and /props."""
    return {
        f"{BASE}/models": ok({"data": [{"id": model_path}]}),
        "http://host:8001/props": ok({
            "model_path": model_path, "total_slots": 4,
            "default_generation_settings": {"n_ctx": 50000},
        }),
    }  # fmt: skip


def llama_cfg(**kw):
    return agent_cfg(
        base_url=BASE, expected_model="Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        model_description="Qwen3.6-35B-A3B, GGUF UD-IQ4_XS", **kw,
    )  # fmt: skip


def test_preflight_records_what_the_server_reports():
    from bench.agent.backends import preflight

    t = MockTransport(gets=llama_server())
    info = preflight(llama_cfg(), transport=t)
    assert info["matched"] and info["models_endpoint"] == [GGUF]
    assert info["props"] == {"model_path": GGUF, "model_alias": None,
                             "n_ctx_per_slot": 50000, "total_slots": 4}  # fmt: skip
    assert [r["url"] for r in t.get_requests] == [f"{BASE}/models", "http://host:8001/props"]
    assert t.requests == []  # no completion was requested


def test_preflight_aborts_on_a_mismatch_with_a_clear_error():
    from bench.agent.backends import preflight

    t = MockTransport(gets=llama_server("/opt/shared/models/gemma4-26b/gemma-4-26B.gguf"))
    with pytest.raises(BackendError, match=r"reports.*gemma-4-26B.gguf.*expected_model.*Qwen3.6"):
        preflight(llama_cfg(), transport=t)


def test_preflight_works_without_props_and_needs_expected_model_and_a_reachable_server():
    from bench.agent.backends import preflight

    only_models = {f"{BASE}/models": ok({"data": [{"id": "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"}]})}
    info = preflight(llama_cfg(), transport=MockTransport(gets=only_models))  # /props -> 404
    assert info["props"] == {}
    with pytest.raises(BackendError, match="expected_model is required"):
        preflight(agent_cfg(), transport=MockTransport())
    down = MockTransport(gets={f"{BASE}/models": HttpResponse(503, {}, "down")})
    with pytest.raises(BackendError, match="attempts failed"):
        preflight(llama_cfg(max_retries=2), transport=down, sleep=lambda s: None)
    assert preflight(agent_cfg("anthropic"), transport=MockTransport()) == {}  # hosted: none


def test_preflight_sends_the_key_but_never_records_it():
    from bench.agent.backends import preflight

    t = MockTransport(gets=llama_server())
    info = preflight(llama_cfg(api_key_env="MY_KEY"), transport=t, environ={"MY_KEY": SECRET})
    assert t.get_requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in json.dumps(info)


# ---- the real runner path (mocked HTTP) and the dry-run guarantee ------------------------------


def runner_args(tmp_path, *extra):
    f = tmp_path / "agent.yaml"
    f.write_text(yaml.safe_dump({"agent": {
        "backend": "openai_compatible", "base_url": BASE, "model": "m-1",
        "expected_model": "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "temperature": 0.6, "top_p": 0.95,
        "top_k": 20, "min_p": 0.0, "model_description": "Qwen3.6-35B-A3B, GGUF UD-IQ4_XS",
    }}))  # fmt: skip
    return [str(ROOT / "configs/exp_toy_1pop.yaml"), "--agent", str(f), "--budget", "60",
            "--out", str(tmp_path / "runs"), "--task", str(TOY), "--seed", "0", *extra]  # fmt: skip


def test_real_runner_runs_the_preflight_and_records_it_in_config_json(tmp_path, capsys):
    t = MockTransport(
        openai_reply("", [("get_task", "{}")]),
        openai_reply("", [("submit", json.dumps({"outcome": "no_solution"}))]),
        gets=llama_server(),
    )
    run_main(runner_args(tmp_path), transport=t)
    assert "preflight ok" in capsys.readouterr().out
    saved = json.loads((tmp_path / "runs/toy_1pop_K2_seed302__m-1__s0/config.json").read_text())
    assert saved["server_preflight"]["models_endpoint"] == [GGUF]
    assert saved["server_preflight"]["props"]["n_ctx_per_slot"] == 50000
    agent = saved["experiment"]["agent"]
    assert (agent["temperature"], agent["top_p"], agent["top_k"], agent["min_p"]) == (
        0.6,
        0.95,
        20,
        0.0,
    )
    assert agent["model_description"] == "Qwen3.6-35B-A3B, GGUF UD-IQ4_XS"
    assert agent["expected_model"] == "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
    assert t.requests[0]["payload"]["top_k"] == 20  # and the sampling reached the request


def test_real_runner_aborts_before_any_episode_on_a_model_mismatch(tmp_path):
    t = MockTransport(gets=llama_server("/opt/shared/models/qwen3.6-27b/Qwen3.6-27B.gguf"))
    with pytest.raises(SystemExit, match="preflight failed"):
        run_main(runner_args(tmp_path), transport=t)
    assert t.requests == [] and not (tmp_path / "runs").exists()  # no completion, no run directory


def test_dry_run_never_uses_the_network_or_the_preflight(tmp_path):
    args = runner_args(tmp_path, "--dry-run")
    run_main(args, transport=NoNetwork())  # any transport use would raise
    assert (tmp_path / "runs/toy_1pop_K2_seed302__dryrun__s0/result.json").exists()
