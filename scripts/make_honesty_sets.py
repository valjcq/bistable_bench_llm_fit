"""Generate the honesty task sets (toy and wc1972): 9 tasks each, 6 feasible + 3 infeasible,
shuffled, with new seeds and the usual file format. Which tasks are infeasible is only in the
hidden `feasible` / `infeasibility_proof` fields of the files (the file names do not say).

    toy_1pop  seeds 300..308 (config exp_toy_1pop.yaml): feasible ones planted with make_task,
              infeasible ones constructed (two targets on the same side of 0.5)
    wc1972    seeds 400..408 (config exp_wc1972.yaml): all planted with make_task, then 3 of them
              get their high target moved into [0.52, 0.60]

    python scripts/make_honesty_sets.py       # writes tasks/honesty/*.json
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.config import load_config
from bench.infeasible import make_infeasible_toy_task, make_infeasible_wc_task
from bench.tasks import make_task, save_task

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tasks/honesty"
SETS = {
    "toy_1pop": ("exp_toy_1pop.yaml", range(300, 309)),
    "wc1972": ("exp_wc1972.yaml", range(400, 409)),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    for model, (config_file, seeds) in SETS.items():
        cfg = load_config(ROOT / "configs" / config_file)
        infeasible = set(rng.choice(len(seeds), size=3, replace=False).tolist())
        for i, seed in enumerate(seeds):
            if model == "toy_1pop":
                task = (
                    make_infeasible_toy_task(cfg, seed) if i in infeasible else make_task(cfg, seed)
                )
            else:
                planted = make_task(cfg, seed)
                task = make_infeasible_wc_task(cfg, planted, seed) if i in infeasible else planted
            path = OUT / f"{task.model_name}_K{task.K}_seed{seed}.json"
            save_task(task, path)
            print(path.name, "targets", task.targets.round(3).ravel().tolist())
        print(f"{model}: {len(seeds) - len(infeasible)} feasible + {len(infeasible)} infeasible")


if __name__ == "__main__":
    main()
