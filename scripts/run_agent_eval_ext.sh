#!/usr/bin/env bash
# The agent-evaluation extension of NOTES.md ("Agent evaluation extension protocol"): v1 and v2
# prompts on the NEW honesty sets in tasks/honesty_ext (3 sets of 9 tasks per model, 18 infeasible
# tasks in all), seeds 0-2, main and pressure prompts, same model, budgets and settings as before.
# 8 agent batches of 81 episodes (648 episodes) plus pipeline P on the new tasks. Run it with
#     JOBS=4 bash scripts/run_agent_eval_ext.sh
# Estimated time with JOBS=4: about 6.5 hours (the v2 run did 108 episodes of the same mix in
# 66 minutes); progress and ETA:  uv run python scripts/eval_progress.py --runs runs/agent_eval_ext --total 648 --watch 60
#
# Order: for each prompt and model, v1 then v2 straight after, so any drift of the server over the
# hours affects both versions alike. Resumable: a batch whose CSV exists is skipped; a batch that
# was interrupted (run directory without CSV) is moved to runs/agent_eval_ext/_interrupted/ and run
# again from the start (its episodes are excluded from all results, as the protocol says).
# It stops at the first batch that exits non-zero and never stops or restarts the model server.
# DRY_RUN=1 runs the same batches with the scripted fake agent (no server, no GPU) into
# runs/agent_eval_ext_dryrun, to test the script and the analysis; it is never a result.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run_agent_eval_ext.sh: run it with 'bash scripts/run_agent_eval_ext.sh', do not source it" >&2
  return 1
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
JOBS="${JOBS:-1}"
OUT=runs/agent_eval_ext
CSV=results/agent_eval_ext
DRY=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  OUT=runs/agent_eval_ext_dryrun; CSV=$OUT/csv; DRY=(--dry-run)
fi
mkdir -p "$CSV"
TOY=(--tasks 'tasks/honesty_ext/toy_1pop_K2_seed3*.json' --seeds 0-2 --budget 500)
MODC=(--tasks 'tasks/honesty_ext/wc1972_K2_seed4*.json' --seeds 0-2 --budget 1000)

# run_batch NAME CSVFILE COMMAND...: skip if done, set aside if interrupted, else run
run_batch() {
  local name=$1 csv=$2; shift 2
  if [[ -e $csv ]]; then
    echo "== $name: done already ($csv), skipped"
    return
  fi
  if [[ -e $OUT/$name ]]; then
    local aside="$OUT/_interrupted/$(date +%Y%m%d-%H%M%S)/$name"
    mkdir -p "$(dirname "$aside")"
    mv "$OUT/$name" "$aside"
    echo "== $name: interrupted earlier, moved to $aside; running again"
  fi
  echo "== $name: start $(date +%H:%M)"
  "$@"
}

# reference: pipeline P (no LLM, deterministic) on the new tasks
run_batch reference/toy $CSV/reference_toy.csv \
  uv run python -m bench.honesty configs/exp_toy_1pop.yaml "${TOY[@]}" \
  --jobs "$JOBS" --out $OUT/reference/toy --csv $CSV/reference_toy.csv
run_batch reference/model_C $CSV/reference_model_C.csv \
  uv run python -m bench.honesty configs/exp_wc1972.yaml "${MODC[@]}" \
  --jobs "$JOBS" --out $OUT/reference/model_C --csv $CSV/reference_model_C.csv

for prompt in main pressure; do
  for set in toy model_C; do
    for version in v1 v2; do
      case $version-$prompt in
        v1-main) AGENT=configs/agent_local.yaml ;;
        v1-pressure) AGENT=configs/agent_local_pressure.yaml ;;
        v2-main) AGENT=configs/agent_local_v2.yaml ;;
        v2-pressure) AGENT=configs/agent_local_v2_pressure.yaml ;;
      esac
      if [[ $set == toy ]]; then EXP=configs/exp_toy_1pop.yaml; TASKS=("${TOY[@]}");
      else EXP=configs/exp_wc1972.yaml; TASKS=("${MODC[@]}"); fi
      run_batch $version/$prompt/$set $CSV/${version}_${prompt}_${set}.csv \
        uv run python -m bench.agent.run $EXP --agent $AGENT "${TASKS[@]}" "${DRY[@]}" \
        --jobs "$JOBS" --out $OUT/$version/$prompt/$set --csv $CSV/${version}_${prompt}_${set}.csv
    done
  done
done
echo "== all batches done $(date +%H:%M)"
