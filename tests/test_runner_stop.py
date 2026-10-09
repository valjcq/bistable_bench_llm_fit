"""The batch safety stop of the agent runner: N harness errors in a row end the batch (nothing is
retried or changed), a good episode in between resets the count."""

import pytest

import bench.agent.run as run


def fake_job(args):
    return {"i": args, "harness_error": args in BAD}


BAD: set[int] = set()


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    BAD.clear()
    monkeypatch.setattr(run, "_job", fake_job)


def test_three_harness_errors_in_a_row_stop_the_batch():
    BAD.update({2, 3, 4})
    rows, stopped = run.run_batch(list(range(10)), jobs=1, max_consecutive_errors=3)
    assert stopped and [r["i"] for r in rows] == [0, 1, 2, 3, 4]  # nothing after the third


def test_a_good_episode_resets_the_count():
    BAD.update({1, 2, 4, 5, 7, 8})
    rows, stopped = run.run_batch(list(range(10)), jobs=1, max_consecutive_errors=3)
    assert not stopped and len(rows) == 10


def test_zero_disables_the_stop():
    BAD.update(range(10))
    rows, stopped = run.run_batch(list(range(10)), jobs=1, max_consecutive_errors=0)
    assert not stopped and len(rows) == 10
