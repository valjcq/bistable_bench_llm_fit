"""Progress and ETA of a running agent evaluation, read from its run directories only.

The agent runner prints nothing until a whole batch (27 episodes) is done, so a long evaluation
gives no sign of life. This script answers "how far along is it, and when will it finish?" from
the outside, without touching the runner or the episodes: an episode directory with a
`result.json` is finished, one without it is still running. The start time is the oldest
`config.json` (written when an episode starts), and the ETA assumes the remaining episodes take,
on average, as long as the finished ones did, with the same number of parallel jobs. It is a rough
estimate: model_C episodes take about twice as long as toy ones, and the slowest episodes come
in no particular order.

    uv run python scripts/eval_progress.py [--runs runs/agent_eval_v2] [--total 108] [--watch 60]
    uv run python scripts/eval_progress.py --runs runs/agent_eval_ext --total 648 --watch 60
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path


def report(runs: Path, total: int) -> str:
    """One status line: finished / running / total, elapsed time and the estimated finish."""
    # agent episodes only, at any depth (runs/<prompt>/<set>/... or runs/<version>/<prompt>/...):
    # pipeline P ("reference", no LLM) and set-aside interrupted batches are not counted
    configs = [
        c
        for c in runs.rglob("config.json")
        if not {"reference", "_interrupted"} & set(c.relative_to(runs).parts)
    ]
    if not configs:
        return f"{runs}: no episode has started yet"
    done = sum((c.parent / "result.json").exists() for c in configs)
    running = len(configs) - done
    start = min(c.stat().st_mtime for c in configs)
    elapsed = time.time() - start
    line = (
        f"{datetime.now():%H:%M} | {done}/{total} done, {running} running | "
        f"elapsed {timedelta(seconds=int(elapsed))}"
    )
    if done >= total:
        line += " | finished"
    elif done:
        remaining = elapsed / done * (total - done)
        finish = datetime.now() + timedelta(seconds=remaining)
        line += f" | ETA {timedelta(seconds=int(remaining))} (about {finish:%H:%M})"
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs/agent_eval_v2"))
    ap.add_argument("--total", type=int, default=108, help="episodes in the whole evaluation")
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds (0: once)")
    args = ap.parse_args()
    while True:
        print(report(args.runs, args.total), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
