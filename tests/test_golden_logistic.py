"""The Wilson-Cowan generalization must leave the logistic models bit-for-bit unchanged.

tests/golden/logistic_outputs.json was written by tests/golden_logistic_gen.py BEFORE the
generalization (checker results, simulations, both objectives, rhs / residual / Jacobian for the
toy and ei_2pop, planted and random parameters). Floats are compared exactly.
"""

import json
from pathlib import Path

from golden_logistic_gen import outputs


def test_logistic_outputs_are_bit_identical_to_the_pre_generalization_golden_file():
    golden = json.loads((Path(__file__).parent / "golden/logistic_outputs.json").read_text())
    now = json.loads(json.dumps(outputs()))  # same JSON round trip as the stored file
    assert now == golden
