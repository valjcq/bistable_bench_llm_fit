"""Evaluation budget: a counter that refuses to go past its limit."""

from __future__ import annotations


class BudgetExhausted(Exception):
    """Raised when an evaluation is requested but the budget is used up."""


class Budget:
    """Counts evaluations. One call to a budgeted Environment method = one evaluation.

    The Environment keeps its Budget private and exposes only read-only counters, so nothing
    outside can reset or refund it.
    """

    def __init__(self, max_evaluations: int) -> None:
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        self._max = int(max_evaluations)
        self._used = 0

    @property
    def total(self) -> int:
        return self._max

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._max - self._used

    def charge(self) -> int:
        """Spend one evaluation and return its 1-based index; raise if none is left.

        The check happens BEFORE any computation, so with a budget of L the first L calls
        succeed and call L+1 raises without doing (or leaking) any work.
        """
        if self._used >= self._max:
            raise BudgetExhausted(f"budget of {self._max} evaluations is used up")
        self._used += 1
        return self._used
