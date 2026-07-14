from __future__ import annotations

import sys
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from export_abides_holdout_audit_bundle import (  # noqa: E402
    GROUPS,
    claim_classification,
)
from verify_abides_holdout_audit_bundle import (  # noqa: E402
    exact_sign_test_independent,
    group_spec,
    holm_adjust_independent,
    t_interval_independent,
)


def test_holdout_groups_are_explicit_and_never_pooled() -> None:
    assert GROUPS == (
        "official_full_policy_MAPPO",
        "official_IPPO",
        "ecology_e1",
        "ecology_e2",
    )
    official_methods, official_contrasts, official_secondary = group_spec(
        "official_IPPO"
    )
    ecology_methods, ecology_contrasts, ecology_secondary = group_spec("ecology_e1")
    assert "brero_threshold_buybox_budget_matched" in official_methods
    assert "brero_threshold_buybox_budget_matched" in official_contrasts
    assert official_secondary == ("brero_threshold_buybox_budget_matched",)
    assert "spread_only_threshold" in ecology_methods
    assert "matched_random" in ecology_contrasts
    assert "matched_random" not in ecology_secondary


def test_independent_t_and_sign_statistics_handle_ties_and_constant_effects() -> None:
    interval = t_interval_independent([1.0, 1.0, 1.0])
    assert interval["mean"] == 1.0
    assert interval["ci_low"] == 1.0
    assert interval["ci_high"] == 1.0
    assert interval["positive"] == 3
    assert interval["p_value_two_sided"] == 0.0

    sign = exact_sign_test_independent([2.0, 1.0, -1.0, 0.0])
    assert sign == {
        "positive": 2,
        "negative": 1,
        "ties": 1,
        "non_ties": 3,
        "p_value_two_sided": 1.0,
    }


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust_independent({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_claim_classification_preserves_inconclusive_positive_group() -> None:
    statistics = {
        group: {
            "contrasts": {
                "matched_random": {
                    "t_interval": {"mean": 0.1, "ci_low": 0.01}
                }
            }
        }
        for group in GROUPS
    }
    statistics["official_IPPO"]["contrasts"]["matched_random"]["t_interval"] = {
        "mean": 0.05,
        "ci_low": -0.2,
    }
    classification = claim_classification(statistics)
    assert classification["official_full_policy_MAPPO"] == (
        "favorable_timing_interval"
    )
    assert classification["official_IPPO"] == (
        "positive_direction_inconclusive_interval"
    )
