"""Generates tests/golden/logistic_outputs.json: exact outputs of the logistic (toy, ei_2pop)
models. Written BEFORE the Wilson-Cowan generalization and compared bit for bit afterwards
(tests/test_golden_logistic.py), so the generalization provably leaves them unchanged.

    uv run python tests/golden_logistic_gen.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench.checker import check, find_stable_states
from bench.config import BudgetConfig, load_config
from bench.env import Environment
from bench.models.rate_network import fixed_point_residual, jacobian, rhs
from bench.simulate import simulate_batch
from bench.tasks import load_task

ROOT = Path(__file__).resolve().parents[1]
CASES = [("toy_1pop", "toy_1pop_K2_seed1"), ("ei_2pop", "ei_2pop_K2_seed1")]


def outputs() -> dict:
    out: dict = {}
    for name, task_name in CASES:
        cfg = replace(load_config(ROOT / f"configs/exp_{name}.yaml"),
                      budget=BudgetConfig(10_000))  # fmt: skip
        task = load_task(ROOT / f"tasks/{task_name}.json")
        model = cfg.model()
        env = Environment.from_task(cfg, task)
        rng = np.random.default_rng(123)
        rows = []
        for i in range(10):
            params = (
                task.planted_params
                if i == 0
                else model.space.from_unit(rng.random(model.space.dim))
            )
            net = model.net(params)
            r = rng.random(model.n_pops)
            x0s = rng.random((3, model.n_pops))
            sims = simulate_batch(model, params, x0s, 300.0, cfg.simulator.sim_config())
            rows.append(
                {
                    "params": params,
                    "rhs": rhs(r, net).tolist(),
                    "residual": fixed_point_residual(r, net).tolist(),
                    "jacobian": jacobian(r, net).tolist(),
                    "check": check(model, params, task.targets, cfg.checker).to_dict(),
                    "stable": [fp.r for fp in find_stable_states(model, params, cfg.checker)],
                    "naive": env.evaluate(params, "naive"),
                    "expert_v2": env.evaluate(params, "expert_v2"),
                    "sims": [
                        [s.final_state, s.settled, s.time_to_settle, s.t_end, s.n_steps]
                        for s in sims
                    ],
                }
            )
        out[name] = rows
    return out


if __name__ == "__main__":
    path = ROOT / "tests/golden/logistic_outputs.json"
    path.write_text(json.dumps(outputs()))
    print("wrote", path, path.stat().st_size, "bytes")
