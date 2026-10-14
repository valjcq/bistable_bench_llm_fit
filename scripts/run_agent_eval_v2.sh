#!/usr/bin/env bash
# The agent evaluation v2 of NOTES.md ("Agent evaluation v2 protocol"): the same model, tasks,
# budgets and seeds as run_agent_eval.sh, with the system_v2 prompts (system_v1 plus one sentence
# saying that scores are not distances to the targets). Main prompt on both sets, then the pressure
# prompt on both sets, seeds 0-2, preflight on. Run it with
#     bash scripts/run_agent_eval_v2.sh
# Pipeline P is not re-run: it uses no prompt and is deterministic, so the v1 reference files
# (results/agent_eval/reference_*.csv) are the reference for v2 as well.
# Outputs go to runs/agent_eval_v2 and results/agent_eval_v2, never to the v1 locations; the script
# refuses to start if either already exists. JOBS as in run_agent_eval.sh. It stops at the first
# batch that exits non-zero and never stops or restarts the model server.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run_agent_eval_v2.sh: run it with 'bash scripts/run_agent_eval_v2.sh', do not source it" >&2
  return 1
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
JOBS="${JOBS:-1}"
OUT=runs/agent_eval_v2
CSV=results/agent_eval_v2
for d in "$OUT" "$CSV"; do
  if [[ -e $d ]]; then
    echo "run_agent_eval_v2.sh: $d already exists; move it away first (nothing was run)" >&2
    exit 1
  fi
done
mkdir -p "$CSV"
TOY=(--tasks 'tasks/honesty/toy_1pop_K2_seed30*.json' --seeds 0-2 --budget 500)
MODC=(--tasks 'tasks/honesty/wc1972_K2_seed40*.json' --seeds 0-2 --budget 1000)

for variant in main pressure; do
  if [[ $variant == main ]]; then AGENT=configs/agent_local_v2.yaml; else AGENT=configs/agent_local_v2_pressure.yaml; fi
  uv run python -m bench.agent.run configs/exp_toy_1pop.yaml --agent $AGENT "${TOY[@]}" \
    --jobs "$JOBS" --out $OUT/$variant/toy --csv $CSV/${variant}_toy.csv
  uv run python -m bench.agent.run configs/exp_wc1972.yaml --agent $AGENT "${MODC[@]}" \
    --jobs "$JOBS" --out $OUT/$variant/model_C --csv $CSV/${variant}_model_C.csv
done
