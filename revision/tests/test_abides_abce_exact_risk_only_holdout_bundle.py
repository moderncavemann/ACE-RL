from __future__ import annotations

import sys
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from verify_abides_abce_exact_risk_only_holdout_v1 import (  # noqa: E402
    ARTIFACT_VERSION,
    classify,
    exact_sign_test_independent,
    t_interval_independent,
)


def test_exact_learner_claim_classification_preserves_the_ippo_boundary() -> None:
    assert classify({"mean": 0.1, "ci_low": 0.01}) == "favorable_timing_interval"
    assert classify({"mean": 0.1, "ci_low": -0.01}) == (
        "positive_direction_inconclusive_interval"
    )
    assert classify({"mean": -0.1, "ci_low": -0.2}) == (
        "nonpositive_timing_direction"
    )


def test_exact_learner_statistics_handle_ties_and_constant_effects() -> None:
    interval = t_interval_independent([1.0, 1.0, 1.0])
    assert interval["mean"] == 1.0
    assert interval["ci_low"] == 1.0
    assert interval["positive"] == 3
    sign = exact_sign_test_independent([2.0, 1.0, -1.0, 0.0])
    assert sign == {
        "positive": 2,
        "negative": 1,
        "ties": 1,
        "non_ties": 3,
        "p_value_two_sided": 1.0,
    }


def test_exact_learner_artifact_name_is_frozen() -> None:
    assert ARTIFACT_VERSION == "abides_abce_exact_risk_only_learner_holdout_v1"
