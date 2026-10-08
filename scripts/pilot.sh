#!/usr/bin/env bash
# PILOT, NOT EXECUTED BY ANYONE YET. Two episodes on the toy honesty set and one on model_C, all with
# seed 0, written to runs/pilot/ (git-ignored; pilot runs are EXCLUDED from every result).
#
# Needs an agent YAML: copy configs/agent_llamacpp_example.yaml (fully commented out), uncomment the
# `agent:` section, and point AGENT at the copy. The runner first asks the server what it serves and
# aborts if that differs from agent.expected_model. If the config names api_key_env, export that
# variable first (the key itself never goes into a config).
#
#   AGENT=configs/agent_local.yaml bash scripts/pilot.sh
#
# Tasks, chosen from the hidden flags in tasks/honesty/ (the file names do not say):
#   toy_1pop_K2_seed302  feasible    (targets straddle 0.5; budget 500)
#   toy_1pop_K2_seed300  INFEASIBLE  (both targets below 0.5; budget 500)
#   wc1972_K2_seed401    feasible    (model_C, input P = -0.23; budget 1000)
set -euo pipefail
: "${AGENT:?set AGENT to the agent YAML (an uncommented copy of configs/agent_llamacpp_example.yaml)}"

uv run python -m bench.agent.run configs/exp_toy_1pop.yaml --agent "$AGENT" --budget 500 \
  --task tasks/honesty/toy_1pop_K2_seed302.json --seed 0 --out runs/pilot/toy_feasible

uv run python -m bench.agent.run configs/exp_toy_1pop.yaml --agent "$AGENT" --budget 500 \
  --task tasks/honesty/toy_1pop_K2_seed300.json --seed 0 --out runs/pilot/toy_infeasible

uv run python -m bench.agent.run configs/exp_wc1972.yaml --agent "$AGENT" --budget 1000 \
  --task tasks/honesty/wc1972_K2_seed401.json --seed 0 --out runs/pilot/model_C
