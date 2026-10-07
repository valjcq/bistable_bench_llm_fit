"""Optimizer wrappers: random search, TPE, CMA-ES, behind one interface.

Nothing in this package imports the model, the checker or the environment (see base.py).
"""

from bench.optimizers.base import HyperSpec, Optimizer, OptimizerSpec, RunResult
from bench.optimizers.cma import CMA
from bench.optimizers.random_search import RandomSearch
from bench.optimizers.tpe import TPE

OPTIMIZERS: dict[str, Optimizer] = {o.spec.name: o for o in (RandomSearch(), TPE(), CMA())}

__all__ = [
    "CMA",
    "OPTIMIZERS",
    "TPE",
    "HyperSpec",
    "Optimizer",
    "OptimizerSpec",
    "RandomSearch",
    "RunResult",
]
