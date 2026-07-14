from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from run_abides_collusion_diagnostic import (  # noqa: E402
    action_for_arm,
    arm_summary,
    event_dealer_index,
    event_is_eligible,
    summarize,
)


def test_event_selection_uses_frozen_boundaries() -> None:
    common = {
        "risk_state": "low",
        "proposed_half_spreads": [6.0] * 10,
        "step": 89,
        "episode_steps": 100,
        "required_risk_state": "low",
        "minimum_mean_half_spread": 6.0,
        "minimum_remaining_steps": 11,
    }
    assert event_is_eligible(**common)
    assert not event_is_eligible(**{**common, "risk_state": "high"})
    assert not event_is_eligible(
        **{**common, "proposed_half_spreads": [5.9] * 10}
    )
    assert not event_is_eligible(**{**common, "step": 90})


def test_dealer_hash_and_action_feasibility_are_deterministic() -> None:
    selected = event_dealer_index(530000, 536000, 7, 10)
    assert selected == event_dealer_index(530000, 536000, 7, 10)
    assert 0 <= selected < 10
    assert action_for_arm(3, "forced_undercut", 7) == (2, True)
    assert action_for_arm(0, "forced_undercut", 7) == (0, False)
    assert action_for_arm(6, "upward_deviation", 7) == (6, False)
    with pytest.raises(ValueError):
        action_for_arm(3, "unknown", 7)


def test_arm_summary_excludes_immediate_reward_from_continuation() -> None:
    rows = [
        {
            "relative_step": step,
            "deviator_reward": float(step + 1),
            "deviator_flow_share": 0.1,
            "aggregate_dealer_reward": 2.0,
            "customer_execution_half_spread": 4.0,
            "rivals_mean_half_spread": 6.0,
            "joint_mean_half_spread": 5.9 if step == 0 else 6.0,
        }
        for step in range(3)
    ]
    result = arm_summary(rows, 6.0)
    assert result["deviator_immediate_reward"] == 1.0
    assert result["deviator_cumulative_continuation_reward"] == 5.0
    assert result["return_to_pre_event_spread_time"] == 1
    assert result["high_spread_recovered_by_h"] == 1


def _synthetic_config(engineering_smoke: bool = False) -> dict:
    return {
        "protocol_version": "test",
        "engineering_smoke": engineering_smoke,
        "required_policy_seed_count": 2,
        "source_policy_seeds": [1, 2],
        "bootstrap_replicates": 20,
        "bootstrap_seed": 7,
        "fallback_wording": "high-spread coordination",
        "estimand_rules": {"minimum_recovery_rate_ci_lower_bound": 0.5},
    }


def _synthetic_result(policy_seed: int, asymmetry: float) -> dict:
    row = {
        "policy_seed": policy_seed,
        "evaluation_seed": 10,
        "event_found": True,
        "forced_undercut_feasible": True,
        "upward_deviation_feasible": True,
        "rival_spread_compression_after_undercut": 0.2,
        "deviator_continuation_flow_loss_after_undercut": 0.2,
        "deviator_continuation_reward_loss_after_undercut": 1.0,
        "high_spread_recovery_rate_after_undercut": 1.0,
        "rival_spread_compression_after_upward": 0.0,
        "deviator_continuation_flow_loss_after_upward": 0.0,
        "deviator_continuation_reward_loss_after_upward": 0.0,
        "flow_loss_asymmetry": asymmetry,
        "reward_loss_asymmetry": asymmetry,
    }
    return {
        "policy_seed": policy_seed,
        "eligible_event_count": 1,
        "event_rows": [row],
        "trajectory_rows": [],
        "integrity": {"all": True},
    }


def test_claim_is_fail_closed_when_upward_asymmetry_fails() -> None:
    results = [_synthetic_result(1, 0.0), _synthetic_result(2, 0.0)]
    summary = summarize(_synthetic_config(), results, [])
    assert not summary["strong_collusion_requirements"][
        "upward_deviation_asymmetry"
    ]
    assert summary["scientific_result"] == "high_spread_coordination_only"
    assert summary["required_wording"] == "high-spread coordination"


def test_smoke_never_emits_a_scientific_result() -> None:
    results = [_synthetic_result(1, 0.2), _synthetic_result(2, 0.2)]
    summary = summarize(_synthetic_config(engineering_smoke=True), results, [])
    assert summary["scientific_result"] == "not_evaluated_engineering_smoke"
