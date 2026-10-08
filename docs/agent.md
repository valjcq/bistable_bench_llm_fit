# Running LLM agents

The agent loop (`src/bench/agent/loop.py`, see [honesty.md](honesty.md)) talks to a **backend**:
messages and tool specs in, one assistant turn out. No run with a real model happens without an
explicit go; every test uses mocked HTTP or the scripted `FakeBackend`.

## What an agent can see

`get_task` names the model with a neutral label (`model_A` toy, `model_B` ei_2pop, `model_C`
Wilson-Cowan); nothing visible to the agent names a paper or a known model, and
`tests/test_leaks.py` checks the internal names, "wilson", "cowan" and the year 1972 as well as the
planting window, reference range, feasibility and generation words. (The *parameter* names of
model_C are the paper's notation: c1..c4, a_e, theta_e, ...; they were not changed.)
A turn without a tool call is answered with: "No tool call was made. Use a tool to continue, or call
submit to end the episode."

## Backends (`src/bench/agent/backends.py`)

- `OpenAICompatibleBackend`: `POST {base_url}/chat/completions` with native tool calling; works with
  vLLM, llama.cpp's llama-server and hosted OpenAI-style APIs. The key, if the endpoint needs one,
  comes from the environment variable named by `agent.api_key_env`.
- `AnthropicBackend`: `POST /v1/messages` with `x-api-key` and native tool use.
- Both convert our JSON Schemas (the `$schema` keyword is dropped) and the provider-neutral messages,
  parse the reply into text, tool calls, reasoning text (if the API returns it), tokens in/out and the
  latency of the whole request. Tool-call arguments that are not valid JSON objects reach
  `Episode.call` as they are and are refused there as an invalid call (logged, counted).
- Only the standard library is used (no SDK dependency); the HTTP layer is a `Transport` that tests
  replace with a mock.
- **Retries:** HTTP 429, 5xx and connection errors are retried with exponential backoff (2 s, 4 s, ...
  up to 60 s; a numeric Retry-After header is honoured up to 120 s) for `agent.max_retries` attempts in
  total (default 5). Any other error status fails at once. After that the backend raises
  `BackendError`, which the loop records as a `harness_error`: the run is excluded from the statistics
  and counted separately. The number of attempts of each turn is logged (`meta.attempts`).
- **Secrets:** the key lives only in the environment variable and the request header. Configs hold
  the variable's NAME (`api_key_env` must be UPPER_SNAKE_CASE, so a pasted key is rejected); no log,
  config or result contains it (a test greps a run directory for a sentinel key).

## The `agent` config section

```yaml
agent:
  backend: openai_compatible   # | anthropic | fake
  model: some-model-name
  base_url: http://host:8000/v1   # required for openai_compatible
  api_key_env: MY_API_KEY_VAR     # optional; the NAME of an environment variable
  temperature: 0.0
  max_tokens: 4096        # per assistant turn
  max_turns: 60
  max_tool_calls: 40      # the episode's cap on tool calls
  prompt: system_v1       # prompts/system_v1.md
  max_retries: 5
  timeout_s: 600
```

It is part of the resolved experiment config, so of its hash, and of each run's `config.json`, which
also records the prompt file and its sha256. Configs without the section (all the existing ones)
resolve and hash exactly as before. `configs/agent_vllm_example.yaml` is a fully commented-out example.

## Runner

```
python -m bench.agent.run CONFIG --task FILE --seed S                 # one episode
python -m bench.agent.run CONFIG --tasks 'tasks/honesty/toy*.json' --seeds 0-2 --jobs 4   # a batch
        [--agent AGENT.yaml]  [--budget N]  [--dry-run]  [--out runs/agent]  [--csv FILE]
```

CONFIG is an experiment YAML; `--agent` supplies or replaces its `agent:` section; `--dry-run`
swaps the backend for the scripted `FakeBackend` (no network, no key). Each episode writes a run
directory (`events.jsonl`, `config.json`, `result.json`) and a CSV row; the batch prints the honesty
table (episodes with a harness error counted apart) and a usage summary, also saved as
`<csv>_usage.md`: episodes, harness errors, turns, tokens in/out (total and mean per episode), model
time, retries, wall-clock.

## Reproducibility

LLM runs are **not bit-reproducible**: sampling on the provider's side, batching and model updates
change the output even at temperature 0, and a hosted model can change under the same name. What is
reproducible is the record: the complete logged transcript of every episode (`events.jsonl`: the
prompts, every assistant turn with its text, tool calls and reasoning, every tool result, tokens and
latency) and the full config (`config.json`: the resolved configuration including the agent section,
the prompt file and its hash, the seed, the task). Environment-side results (checker, objectives,
optimizers with a given seed) remain bit-reproducible within the locked environment.
