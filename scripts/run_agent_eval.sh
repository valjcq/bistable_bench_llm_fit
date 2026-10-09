#!/usr/bin/env bash
# The agent evaluation of NOTES.md ("Agent evaluation protocol"): pipeline P, then the main prompt
# on both sets, then the pressure prompt on both sets, seeds 0-2, preflight on. Run it with
#     bash scripts/run_agent_eval.sh
# JOBS (default 1) is the number of parallel episodes; use more than 1 only if the preflight
# reported >= 2 slots with >= 64k context each (and then at most the number of slots). The script
# stops at the first batch that exits non-zero (preflight failed, or 3 harness errors in a row)
# and changes nothing to make it pass. It never stops or restarts the model server.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run_agent_eval.sh: run it with 'bash scripts/run_agent_eval.sh', do not source it" >&2
  return 1
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
JOBS="${JOBS:-1}"
OUT=runs/agent_eval
TOY=(--tasks 'tasks/honesty/toy_1pop_K2_seed30*.json' --seeds 0-2 --budget 500)
MODC=(--tasks 'tasks/honesty/wc1972_K2_seed40*.json' --seeds 0-2 --budget 1000)

# reference: pipeline P (no LLM), same tasks, budgets and seeds
uv run python -m bench.honesty configs/exp_toy_1pop.yaml "${TOY[@]}" \
  --out $OUT/reference/toy --csv results/agent_eval/reference_toy.csv
uv run python -m bench.honesty configs/exp_wc1972.yaml "${MODC[@]}" \
  --out $OUT/reference/model_C --csv results/agent_eval/reference_model_C.csv

for variant in main pressure; do
  if [[ $variant == main ]]; then AGENT=configs/agent_local.yaml; else AGENT=configs/agent_local_pressure.yaml; fi
  uv run python -m bench.agent.run configs/exp_toy_1pop.yaml --agent $AGENT "${TOY[@]}" \
    --jobs "$JOBS" --out $OUT/$variant/toy --csv results/agent_eval/${variant}_toy.csv
  uv run python -m bench.agent.run configs/exp_wc1972.yaml --agent $AGENT "${MODC[@]}" \
    --jobs "$JOBS" --out $OUT/$variant/model_C --csv results/agent_eval/${variant}_model_C.csv
done
