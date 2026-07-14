from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.interventions.abides_literature_baselines import (  # noqa: E402
    BudgetMetrics,
    TemporaryCommitmentController,
    brero_threshold_decision,
    brero_threshold_eligibility,
    budget_match_status,
    calibrate_domain_baselines,
    calibrate_temporary_commitment,
    eligible_routing_weights,
    routing_total_variation,
    soft_rank_routing_weights,
)


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_brero_candidate_family_is_frozen_and_complete() -> None:
    config = load_config("abides_domain_baseline_calibration_v1.json")
    brero = config["brero_threshold_buybox"]
    assert brero["order_statistic_k_candidates"] == list(range(1, 11))
    assert brero["margin_tick_candidates"] == [0, 1]
    assert brero["literature_form"] == {
        "order_statistic_k": 1,
        "margin_ticks": 0,
    }
    assert brero["empty_rule"] == "include_all_narrowest_ties"


def test_calibration_forbids_efficacy_fields() -> None:
    config = load_config("abides_domain_baseline_calibration_v1.json")
    forbidden = set(config["forbidden_calibration_fields"])
    assert {
        "low_execution_half_spread",
        "mean_reward",
        "dealer_reward_units_per_episode",
        "customer_cost",
    }.issubset(forbidden)
    assert config["selection_rule"].startswith("minimum_normalized_budget_distance")


def test_literature_mapping_precedes_implementation() -> None:
    mapping = REVISION_ROOT / "notes" / "literature_baseline_mapping_v1.md"
    text = mapping.read_text(encoding="utf-8")
    assert "arXiv:2202.07106v2" in text
    assert "arXiv:2303.02576v3" in text
    assert "ABIDES adaptation" in text
    assert "not an exact reproduction" in text


def test_threshold_eligibility_uses_order_statistic_and_keeps_ties() -> None:
    threshold, eligible = brero_threshold_eligibility(
        [3, 1, 2, 2, 5], order_statistic_k=2, margin_ticks=0
    )
    assert threshold == 2.0
    assert eligible == (1, 2, 3)

    threshold, eligible = brero_threshold_eligibility(
        [7, 7, 7], order_statistic_k=1, margin_ticks=0
    )
    assert threshold == 7.0
    assert eligible == (0, 1, 2)


def test_brero_routing_is_deterministic_valid_and_demand_conserving() -> None:
    first = brero_threshold_decision([4, 2, 2, 6], 1, 0, 0.5)
    second = brero_threshold_decision([4, 2, 2, 6], 1, 0, 0.5)
    assert first == second
    assert first.eligible_indices == (1, 2)
    assert np.all(np.asarray(first.routing_weights) >= 0.0)
    assert sum(first.routing_weights) == pytest.approx(1.0)
    assert sum(20.0 * value for value in first.routing_weights) == pytest.approx(20.0)
    assert first.routing_total_variation > 0.0


def test_latent_risk_cannot_change_public_quote_rule() -> None:
    public_quotes = [5, 2, 3, 2]
    low_risk = brero_threshold_decision(public_quotes, 2, 1, 0.5)
    high_risk = brero_threshold_decision(public_quotes, 2, 1, 0.5)
    assert low_risk == high_risk
    assert "risk" not in low_risk.parameters


def test_mechanism_off_is_exact_uniform_routing() -> None:
    weights = eligible_routing_weights(4, [0, 1], 0.5, enabled=False)
    assert weights.tolist() == [0.25, 0.25, 0.25, 0.25]
    assert routing_total_variation(weights) == 0.0


def test_soft_rank_weights_are_valid_and_off_is_uniform() -> None:
    active = soft_rank_routing_weights([1, 2, 3, 4], 0.8, 4.0, 0.5)
    off = soft_rank_routing_weights([1, 2, 3, 4], 0.8, 4.0, 0.5, enabled=False)
    assert np.all(active >= 0.0)
    assert float(np.sum(active)) == pytest.approx(1.0)
    assert routing_total_variation(active) > 0.0
    assert off.tolist() == [0.25, 0.25, 0.25, 0.25]


def synthetic_calibration_records() -> list[dict]:
    records = []
    for episode in range(2):
        for step in range(10):
            records.append(
                {
                    "episode_id": f"episode-{episode}",
                    "step": step,
                    "half_spreads": [1, 2, 3, 4, 5, 6, 7, 4, 5, 6],
                    "risk_signal_strength": 0.6 if step in (4, 9) else 0.0,
                    "lagged_contract_severity": step / 9.0,
                    "latent_risk_state": "low" if step % 2 else "high",
                }
            )
    return records


def test_calibration_uses_budget_only_and_is_deterministic() -> None:
    config = load_config("abides_domain_baseline_calibration_v1.json")
    first = calibrate_domain_baselines(synthetic_calibration_records(), config)
    second = calibrate_domain_baselines(synthetic_calibration_records(), config)
    assert first == second
    assert first["selection_uses_efficacy_outcomes"] is False
    assert len(first["brero_threshold_buybox_budget_matched"]["candidates"]) == 20
    assert len(first["spread_only_budget_matched"]["candidates"]) == 5 * 10 * 3


def test_calibration_rejects_any_efficacy_field() -> None:
    config = load_config("abides_domain_baseline_calibration_v1.json")
    records = synthetic_calibration_records()
    records[0]["mean_reward"] = 123.0
    with pytest.raises(ValueError, match="forbidden efficacy fields"):
        calibrate_domain_baselines(records, config)


def test_budget_match_tolerance_does_not_promote_unmatched_candidate() -> None:
    reference = BudgetMetrics(0.2, 0.1, 0.2, 100)
    candidate = BudgetMetrics(0.25, 0.2, 0.3, 100)
    status, ratio = budget_match_status(candidate, reference, 0.02, 0.9, 1.1)
    assert status == "unmatched"
    assert ratio == pytest.approx(2.0)


def test_frozen_passive_calibration_preserves_unmatched_external_result() -> None:
    artifact = load_config("abides_domain_baseline_calibration_v1_artifact.json")
    selection = artifact["selection"]
    assert selection["record_count"] == 20000
    assert selection["episode_count"] == 200
    assert selection["selection_uses_efficacy_outcomes"] is False
    assert (
        selection["spread_only_budget_matched"]["selected"]["budget_match_status"]
        == "matched"
    )
    assert (
        selection["brero_threshold_buybox_budget_matched"]["selected"][
            "budget_match_status"
        ]
        == "unmatched"
    )


def test_temporary_commitment_uses_lagged_public_quotes_and_deterministic_ties() -> None:
    controller = TemporaryCommitmentController(n_agents=4)
    for _ in range(10):
        assert not controller.decide([5, 5, 5, 5], 0.0).active

    takeup = controller.decide([2, 2, 4, 5], 0.75)
    assert takeup.active
    assert takeup.takeup
    assert takeup.reference_half_spread == 5.0
    assert takeup.dealer_index == 1
    assert takeup.frozen_half_spread == 2.0
    assert takeup.remaining_periods_including_current == 5
    assert takeup.routing_total_variation > 0.0
    assert sum(takeup.routing_weights) == pytest.approx(1.0)


def test_temporary_commitment_freezes_quote_for_exact_duration_without_overlap() -> None:
    controller = TemporaryCommitmentController(n_agents=4, duration=5)
    for _ in range(10):
        controller.decide([5, 5, 5, 5], 0.0)
    first = controller.decide([2, 4, 4, 4], 0.0)
    continuation = [
        controller.decide([7, 1, 7, 7], 0.0) for _ in range(4)
    ]
    assert first.dealer_index == 0
    assert [row.dealer_index for row in continuation] == [0, 0, 0, 0]
    assert [row.frozen_half_spread for row in continuation] == [2.0] * 4
    assert [row.remaining_periods_including_current for row in continuation] == [
        4,
        3,
        2,
        1,
    ]
    expired = controller.decide([7, 7, 7, 7], 0.0)
    assert not expired.active
    assert expired.dealer_index is None
    assert expired.routing_weights == (0.25, 0.25, 0.25, 0.25)


def test_temporary_commitment_calibration_grid_is_frozen_and_budget_only() -> None:
    config = load_config("abides_temporary_commitment_calibration_v1.json")
    mechanism = config["banerjee_temporary_commitment"]
    assert mechanism["literature_form"] == {
        "reference_window": 10,
        "qualification_steps": 3,
        "protected_fraction": 0.5,
        "duration": 5,
    }
    assert mechanism["protected_fraction_candidates"] == [0.1, 0.25, 0.5, 0.75]
    assert mechanism["duration_candidates"] == [1, 3, 5, 10, 20]
    formal = load_config("abides_temporary_commitment_benchmark_v1.json")
    assert formal["freeze_status"] == "frozen_after_passive_calibration"
    assert formal["budget_matched_parameters"]["budget_match_status"] == "unmatched"


def test_temporary_commitment_calibration_is_deterministic() -> None:
    config = load_config("abides_temporary_commitment_calibration_v1.json")
    records = []
    for episode in range(2):
        for step in range(30):
            spreads = [5.0] * 10
            if step in (10, 20):
                spreads[0] = 2.0
            records.append(
                {
                    "episode_id": f"episode-{episode}",
                    "step": step,
                    "half_spreads": spreads,
                    "risk_signal_strength": 0.6 if step % 5 == 0 else 0.0,
                    "lagged_contract_severity": 0.5,
                }
            )
    first = calibrate_temporary_commitment(records, config)
    second = calibrate_temporary_commitment(records, config)
    assert first == second
    assert first["selection_uses_efficacy_outcomes"] is False
    assert len(first["budget_matched"]["candidates"]) == 20
    assert first["record_count"] == 60


def test_frozen_temporary_commitment_selection_matches_passive_artifact() -> None:
    artifact_path = (
        REVISION_ROOT
        / "configs"
        / "abides_temporary_commitment_calibration_v1_artifact.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    formal = load_config("abides_temporary_commitment_benchmark_v1.json")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert formal["formal_execution_allowed"]
    assert formal["calibration_artifact_sha256"] == digest
    selected = artifact["selection"]["budget_matched"]["selected"]
    assert selected["budget_match_status"] == "unmatched"
    for key, value in selected["parameters"].items():
        assert formal["budget_matched_parameters"][key] == value
