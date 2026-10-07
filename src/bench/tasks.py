"""Task generation by PLANTING: build a task whose solution is known to exist.

A task is K target states. To guarantee that it is solvable we first draw parameters at
random, find the stable states they actually have, and use K of those states (observed
populations only) as the targets. The parameters that produced them are stored in the Task as
`planted_params`, for analysis only: the Environment is built from `targets` alone.

Tasks are generated once, saved to tasks/*.json and loaded by runs (see `main`), so every run
of an experiment sees exactly the same tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.checker import check, find_stable_states
from bench.config import ExperimentConfig, load_config


class TaskGenerationError(RuntimeError):
    """Raised when no valid planted task was found within the attempt cap."""


@dataclass(frozen=True)
class Task:
    model_name: str
    observed: tuple[str, ...]
    seed: int
    K: int
    targets: np.ndarray  # (K, n_observed), sorted by increasing mean rate
    planted_params: dict[str, float]  # HIDDEN: analysis only, never given to the environment
    generation_hash: str  # ExperimentConfig.generation_hash() at generation time
    n_attempts: int  # random draws needed
    rejections: dict[str, int]  # why draws were rejected (diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model": self.model_name,
            "observed": list(self.observed),
            "seed": self.seed,
            "K": self.K,
            "targets": self.targets.tolist(),
            "planted_params": self.planted_params,
            "generation_hash": self.generation_hash,
            "n_attempts": self.n_attempts,
            "rejections": self.rejections,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            model_name=d["model"],
            observed=tuple(d["observed"]),
            seed=int(d["seed"]),
            K=int(d["K"]),
            targets=np.array(d["targets"], dtype=float),
            planted_params={k: float(v) for k, v in d["planted_params"].items()},
            generation_hash=d["generation_hash"],
            n_attempts=int(d["n_attempts"]),
            rejections=dict(d["rejections"]),
        )


def save_task(task: Task, path: str | Path) -> None:
    """Write a task as JSON. The file holds the HIDDEN planted params: never show it to an agent."""
    Path(path).write_text(json.dumps(task.to_dict(), indent=2) + "\n")


def load_task(path: str | Path) -> Task:
    return Task.from_dict(json.loads(Path(path).read_text()))


def _separated(rows: list[np.ndarray] | np.ndarray, min_sep: float) -> bool:
    """True if every pair of target rows differs by more than min_sep (max abs difference)."""
    rows = np.asarray(rows)
    return all(
        np.abs(rows[i] - rows[j]).max() > min_sep
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
    )


def _pick_separated(
    candidates: list[np.ndarray], K: int, min_sep: float, rng: np.random.Generator
) -> list[np.ndarray] | None:
    """Choose K candidates pairwise farther than min_sep: random order, greedy, seeded."""
    chosen: list[np.ndarray] = []
    for i in rng.permutation(len(candidates)):
        if all(np.abs(candidates[i] - c).max() > min_sep for c in chosen):
            chosen.append(candidates[i])
            if len(chosen) == K:
                return chosen
    return None


def make_task(config: ExperimentConfig, seed: int | None = None, K: int | None = None) -> Task:
    """Plant a task: draw parameters until they have >= K usable stable states.

    Each draw goes through cheap-to-expensive filters, so the full checker only runs on
    promising candidates (only ~0.3% of random ei_2pop draws qualify, so most draws must be
    rejected cheaply):
      1. PREFILTER: find stable states with the checker's multi-start search using only
         `task.prefilter_starts` starts; keep those whose observed rates lie in the target
         window; need at least K of them. (A cheap search may miss states: that only costs
         acceptance rate, never correctness, because step 4 re-verifies everything.)
      2. pick K candidates pairwise separated by more than min_separation_tol_factor * tol;
      3. add Gaussian noise (task.noise), then re-check window and separation;
      4. full check of the planted params against the final targets: must succeed.
    Same (config, seed, K) -> identical task: all randomness comes from default_rng(seed).
    Targets are returned sorted by increasing mean rate (the naive objective pairs them with
    initial conditions in that order).
    """
    tc = config.task
    seed = tc.seed if seed is None else seed
    K = tc.K if K is None else K
    model = config.model()
    cheap = replace(config.checker, n_starts=tc.prefilter_starts)
    min_sep = tc.min_separation_tol_factor * tc.tol
    rng = np.random.default_rng(seed)
    rejected: Counter[str] = Counter()

    for attempt in range(1, tc.max_attempts + 1):
        params = model.space.from_unit(rng.random(model.space.dim))
        r_max_obs = model.net(params).r_max[model.observed_idx]
        lo, hi = tc.target_window[0] * r_max_obs, tc.target_window[1] * r_max_obs

        def in_window(rows: np.ndarray, lo: np.ndarray = lo, hi: np.ndarray = hi) -> bool:
            return bool(np.all((rows >= lo) & (rows <= hi)))

        stable = find_stable_states(model, params, cheap)
        candidates = [np.array(fp.observed) for fp in stable if in_window(np.array(fp.observed))]
        if len(candidates) < K:
            rejected["fewer_than_K_stable_states_in_window"] += 1
            continue
        chosen = _pick_separated(candidates, K, min_sep, rng)
        if chosen is None:
            rejected["targets_too_close"] += 1
            continue
        targets = np.array(chosen)
        if tc.noise > 0:
            targets = targets + rng.normal(0.0, tc.noise, targets.shape)
            if not (in_window(targets) and _separated(targets, min_sep)):
                rejected["noisy_targets_outside_window_or_too_close"] += 1
                continue
        targets = targets[np.argsort(targets.mean(axis=1), kind="stable")]
        if not check(model, params, targets, config.checker).success:
            rejected["planted_params_fail_full_check"] += 1
            continue
        return Task(
            model_name=model.name,
            observed=model.observed,
            seed=seed,
            K=K,
            targets=targets,
            planted_params=dict(params),
            generation_hash=config.generation_hash(),
            n_attempts=attempt,
            rejections=dict(rejected),
        )

    raise TaskGenerationError(
        f"no valid task for model '{model.name}' (K={K}, seed={seed}) in {tc.max_attempts} "
        f"random draws. Rejections: {dict(rejected)}. Try a larger task.max_attempts, a wider "
        f"task.target_window, a smaller K or min_separation_tol_factor, or different bounds."
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: generate task files once, e.g.
    python -m bench.tasks configs/exp_ei_2pop.yaml --seeds 0 1 2 --out tasks
    """
    ap = argparse.ArgumentParser(description="Generate planted tasks and save them as JSON.")
    ap.add_argument("config", help="experiment YAML")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--K", type=int, default=None, help="override task.K")
    ap.add_argument("--out", default="tasks", help="output directory")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        task = make_task(config, seed=seed, K=args.K)
        path = out / f"{task.model_name}_K{task.K}_seed{seed}.json"
        save_task(task, path)
        print(f"{path}  attempts={task.n_attempts}  targets={task.targets.round(3).tolist()}")


if __name__ == "__main__":
    main()
