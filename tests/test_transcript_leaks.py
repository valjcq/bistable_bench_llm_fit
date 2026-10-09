"""Leak check on a whole recorded episode (not only on single tool outputs).

tests/test_leaks.py collects what the tools can return. This test plays a full scripted episode on
model_C (the model whose internal names are paper notation: c1..c4, a_e, theta_e, a_i, theta_i,
and the input P) through the agent loop and the honesty runner, then checks everything an agent
could have seen of it:

  * every message handed to the backend, in every turn (system prompt, first message, the agent's
    own calls, all tool results, the tool specs), and
  * every line of events.jsonl that is agent-visible: the assistant turns and the lines of the
    tools the agent called by name (get_task, check_bistability, ...).

Internal names may appear only in the harness/environment lines of events.jsonl: the optimizer's
and the environment's own evaluations (the inner `check` calls that a tool makes on the model with
the model's real parameter names). The test also asserts that those lines DO carry internal names,
so it cannot pass vacuously if the mapping were silently removed from the log as well.
"""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from bench.agent.fake import FakeBackend, last_tool_result
from bench.agent.loop import AssistantTurn, ToolCall, load_system_prompt, run_agent
from bench.config import BudgetConfig, load_config
from bench.honesty import run_honesty_episode
from bench.logging import read_events
from bench.tasks import load_task

ROOT = Path(__file__).resolve().parents[1]
HONESTY = ROOT / "tasks/honesty"
FORBIDDEN_WORDS = (
    "wilson", "cowan", "wc1972", "toy_1pop", "ei_2pop", "wc_shifted", "model card",
    "window", "reference", "feasib", "plant", "generated", "proof", "hidden",
)  # fmt: skip
INTERNAL_NAMES = re.compile(r"\b(c[1-4]|a_e|theta_e|a_i|theta_i)\b")
YEAR = re.compile(r"(?<![0-9.])1972(?![0-9])")
STANDALONE = re.compile(r'"[PQ]"')


def leaks(text: str) -> list[str]:
    """What in `text` an agent must never see (internal names, the paper, the year, P or Q)."""
    found = [w for w in FORBIDDEN_WORDS if w in text.lower()]
    found += [m.group(0) for m in INTERNAL_NAMES.finditer(text)]
    found += [m.group(0) for m in YEAR.finditer(text)]
    found += [m.group(0) for m in STANDALONE.finditer(text)]
    return found


def turn(*calls):
    """A scripted turn; ids are call_1, ... (an id such as c1 would look like a parameter name)."""
    return AssistantTurn(
        tool_calls=[ToolCall(f"call_{i}", n, a) for i, (n, a) in enumerate(calls, start=1)]
    )


@pytest.fixture(scope="module")
def episode_run(tmp_path_factory):
    """A scripted model_C episode: tools of every kind, an invalid call, then a submission."""
    out = tmp_path_factory.mktemp("transcript")
    cfg = replace(load_config(ROOT / "configs/exp_wc1972.yaml"), budget=BudgetConfig(120))
    task = load_task(HONESTY / "wc1972_K2_seed401.json")

    def names(messages):
        return [p["name"] for p in last_tool_result(messages, 0)["free_parameters"]]

    def run(messages):
        return turn(("run_optimizer", {"name": "cma", "sim_budget": 40, "objective": "expert_v2"}))

    def inspect(messages):
        best = last_tool_result(messages)["best_params"]
        bad = {**best, names(messages)[0]: 1e9}  # out of bounds: an error message is agent-visible
        return turn(
            ("check_bistability", {"params": best}),
            ("check_bistability", {"params": bad}),
            ("get_run_results", {"run_id": "run1", "fields": ["history", "best_params"]}),
            ("simulate", {"params": best, "init_state": [0.1, 0.1], "T": 20}),
            ("get_budget", {}),
            ("list_optimizers", {}),
        )

    def submit(messages):
        best = last_tool_result(messages, 2)["best_params"]
        return turn(("submit", {"outcome": "solution", "params": best, "justification": "ok"}))

    backend = FakeBackend([turn(("get_task", {})), run, inspect, submit])
    prompt = load_system_prompt("system_v1")
    run_honesty_episode(
        cfg, task, "wc1972_K2_seed401", 0, out, "fake", 40,
        play=lambda episode: run_agent(episode, backend, prompt, max_turns=10),
    )  # fmt: skip
    (run_dir,) = list(out.iterdir())
    return backend, read_events(run_dir / "events.jsonl"), run_dir


def test_messages_sent_to_the_backend_contain_no_internal_name(episode_run):
    backend, _, _ = episode_run
    assert len(backend.calls) == 4
    for messages, tools in backend.calls:
        assert leaks(json.dumps(messages) + json.dumps(tools)) == []


def test_agent_visible_event_lines_contain_no_internal_name(episode_run):
    _, events, _ = episode_run
    # the line a tool call itself writes has a call_id; the environment events it triggers (named
    # evaluate, check, simulate, ... and carrying the model's real parameter names) have only a
    # parent_call_id
    visible = [e for e in events if e["role"] == "assistant" or e["call_id"] is not None]
    assert {e["name"] for e in visible} >= {"get_task", "check_bistability", "assistant_turn"}
    for e in visible:
        assert leaks(json.dumps(e)) == [], e["name"]


def test_internal_names_appear_only_in_environment_lines(episode_run):
    _, events, _ = episode_run
    others = [
        e for e in events
        if e["role"] not in ("assistant", "harness") and e["call_id"] is None
    ]  # fmt: skip
    # the environment's own `check` lines and the optimizer's evaluations name the real parameters
    assert any(INTERNAL_NAMES.search(json.dumps(e)) for e in others)
    # and the harness lines (prompts, nudges, ends) never do
    for e in events:
        if e["role"] == "harness":
            assert leaks(json.dumps(e)) == [], e["name"]
