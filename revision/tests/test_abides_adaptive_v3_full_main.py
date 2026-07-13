from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO = REVISION_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from experiments.run_abides_mappo_adaptive_v3_full_main import (  # noqa: E402
    EVALUATED_METHODS,
    aggregate,
    convergence as response_convergence,
    spread_only_decision,
)
from experiments.run_abides_mappo_adaptive_v3_response import (  # noqa: E402
    market_config,
)
from experiments.run_abides_mappo_full_main import protocol_fingerprint  # noqa: E402
from experiments.train_abides_mappo_base_v3 import (  # noqa: E402
    fixed_horizon_diagnostic,
    summarize_base_rows,
)
from src.interventions.abides_gate import ABIDESPacedSignalController  # noqa: E402


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_full_main_seed_splits_and_base_fingerprints_are_frozen() -> None:
    pairs = (
        ("primary", 20),
        ("fast_risk", 10),
    )
    for ecology, expected_count in pairs:
        base = load_config(f"abides_mappo_adaptive_v3_full_main_{ecology}_base_v1.json")
        response = load_config(f"abides_mappo_adaptive_v3_full_main_{ecology}_v1.json")
        assert len(base["base_training_seeds"]) == expected_count
        assert len(response["policy_seeds"]) == expected_count
        assert base["base_training_seeds"] == response["policy_seeds"]
        assert set(base["base_training_seeds"]).isdisjoint(
            base["base_fixed_evaluation_seeds"]
        )
        assert set(response["policy_seeds"]).isdisjoint(response["evaluation_seeds"])
        assert set(base["base_fixed_evaluation_seeds"]).isdisjoint(
            response["evaluation_seeds"]
        )
        assert protocol_fingerprint(base) == response["base_protocol_fingerprint"]
        assert tuple(response["methods"]) == EVALUATED_METHODS


def test_v4_formal_primary_is_fresh_and_keeps_validated_response_protocol() -> None:
    development = load_config("abides_mappo_adaptive_v4_base_development_v1.json")
    base = load_config("abides_mappo_adaptive_v4_full_main_primary_base_v1.json")
    response = load_config("abides_mappo_adaptive_v4_full_main_primary_v1.json")
    v3_response = load_config("abides_mappo_adaptive_v3_full_main_primary_v1.json")

    assert base["base_training_mode"] == "fixed_horizon"
    assert base["base_max_episodes"] == 384
    assert len(base["base_training_seeds"]) == 20
    assert base["base_training_seeds"] == response["policy_seeds"]
    assert protocol_fingerprint(base) == response["base_protocol_fingerprint"]
    assert tuple(response["methods"]) == EVALUATED_METHODS

    development_seeds = set(development["base_training_seeds"]) | set(
        development["base_fixed_evaluation_seeds"]
    )
    formal_seeds = (
        set(base["base_training_seeds"])
        | set(base["base_fixed_evaluation_seeds"])
        | set(response["evaluation_seeds"])
    )
    assert development_seeds.isdisjoint(formal_seeds)
    assert set(base["base_training_seeds"]).isdisjoint(
        base["base_fixed_evaluation_seeds"]
    )
    assert set(base["base_training_seeds"]).isdisjoint(response["evaluation_seeds"])
    assert set(base["base_fixed_evaluation_seeds"]).isdisjoint(
        response["evaluation_seeds"]
    )

    frozen_response_fields = (
        "docker_image",
        "episode_steps",
        "methods",
        "response_episodes",
        "response_episodes_per_update",
        "response_evaluation_start",
        "response_evaluation_interval",
        "response_required_stable_evaluations",
        "response_convergence_path_count",
        "response_maximum_fixed_panel_spread_range",
        "learning_rate",
        "entropy_coefficient",
        "reward_scale",
        "max_activation_rate",
        "routing_allocation_mode",
        "routing_target_kappa",
        "routing_routable_fraction",
        "market",
        "gate_artifact",
        "gate_artifact_sha256",
        "minimum_positive_seed_rate",
        "minimum_failure_spread",
        "maximum_reward_degradation_fraction",
        "maximum_high_risk_activation_rate",
        "minimum_threshold_high_risk_activation_gap",
    )
    for field in frozen_response_fields:
        assert response[field] == v3_response[field]


def test_v5_convergence_veto_is_estimand_aligned_and_fresh() -> None:
    v4_base = load_config("abides_mappo_adaptive_v4_full_main_primary_base_v1.json")
    v4_response = load_config("abides_mappo_adaptive_v4_full_main_primary_v1.json")
    v5_base = load_config("abides_mappo_adaptive_v5_full_main_primary_base_v1.json")
    v5_response = load_config("abides_mappo_adaptive_v5_full_main_primary_v1.json")

    assert protocol_fingerprint(v5_base) == v5_response["base_protocol_fingerprint"]
    assert set(v4_base["base_training_seeds"]).isdisjoint(
        v5_base["base_training_seeds"]
    )
    assert set(v4_base["base_fixed_evaluation_seeds"]).isdisjoint(
        v5_base["base_fixed_evaluation_seeds"]
    )
    assert set(v4_response["evaluation_seeds"]).isdisjoint(
        v5_response["evaluation_seeds"]
    )
    assert v5_response["response_convergence_methods"] == [
        "no_intervention",
        "risk_aware",
        "matched_random",
        "spread_only_threshold",
        "no_routing",
    ]

    allowed_changes = {
        "protocol_version",
        "policy_seeds",
        "evaluation_seeds",
        "base_output_dir",
        "base_protocol_fingerprint",
        "response_convergence_methods",
        "response_environment_seed_namespace",
        "response_action_namespace",
        "evaluation_action_namespace",
        "output_dir",
    }
    for field in set(v4_response) | set(v5_response):
        if field not in allowed_changes:
            assert v5_response[field] == v4_response[field]


def test_v5_convergence_reports_but_does_not_veto_always_on_drift() -> None:
    v4 = load_config("abides_mappo_adaptive_v4_full_main_primary_v1.json")
    v5 = load_config("abides_mappo_adaptive_v5_full_main_primary_v1.json")
    history = []
    for episode, always_on_spread in ((48, 5.30), (72, 5.20), (96, 5.10)):
        history.append(
            {
                "completed_episodes": episode,
                "evaluation": {
                    "methods": {
                        method: {
                            "low_execution_half_spread": (
                                always_on_spread if method == "always_on" else 6.3
                            )
                        }
                        for method in EVALUATED_METHODS
                    }
                },
            }
        )

    legacy = response_convergence(history, v4)
    aligned = response_convergence(history, v5)
    assert not legacy["stable"]
    assert aligned["stable"]
    assert not aligned["method_stability"]["always_on"]
    assert "always_on" not in aligned["required_methods"]


def test_fast_risk_ecology_preserves_stationary_high_risk_share() -> None:
    base = load_config("abides_mappo_adaptive_v3_full_main_primary_base_v1.json")
    fast = load_config("abides_mappo_adaptive_v3_full_main_fast_risk_base_v1.json")
    base_market = base["market"]
    fast_market = fast["market"]
    base_share = base_market["p_low_to_high"] / (
        base_market["p_low_to_high"] + base_market["p_high_to_low"]
    )
    fast_share = fast_market["p_low_to_high"] / (
        fast_market["p_low_to_high"] + fast_market["p_high_to_low"]
    )
    assert fast_market["p_low_to_high"] == 2 * base_market["p_low_to_high"]
    assert fast_market["p_high_to_low"] == 2 * base_market["p_high_to_low"]
    assert abs(base_share - fast_share) < 1e-12


def test_market_config_carries_predeclared_ecology() -> None:
    config = load_config("abides_mappo_adaptive_v3_full_main_fast_risk_v1.json")
    market = market_config(config)
    assert market["episode_steps"] == 100
    assert market["p_low_to_high"] == 0.08
    assert market["p_high_to_low"] == 0.20
    assert market["routing_allocation_mode"] == "soft_rank"
    assert market["routing_routable_fraction"] == 0.50


def test_spread_only_control_is_pre_action_and_lagged() -> None:
    observations = np.zeros((10, 6), dtype=float)
    observations[:, 1] = 1.0
    state = {
        "observations": observations.tolist(),
        "info": {"public_volatility": 0.0, "flow_shares": [0.1] * 10},
    }
    high = spread_only_decision(ABIDESPacedSignalController(100, target_rate=0.35))
    high_schedule = [high(state, step) for step in range(100)]
    assert sum(value > 0.0 for value in high_schedule) == 35
    observations[:, 1] = 0.0
    state["observations"] = observations.tolist()
    low = spread_only_decision(ABIDESPacedSignalController(100, target_rate=0.35))
    assert all(low(state, step) == 0.0 for step in range(100))


def test_full_main_aggregate_accepts_mechanism_valid_fixture() -> None:
    config = load_config("abides_mappo_adaptive_v3_full_main_primary_v1.json")

    def metrics(spread, reward, activation, high_activation, targeted):
        return {
            "low_execution_half_spread": spread,
            "mean_reward": reward,
            "mean_volume": 20.0,
            "activation_rate": activation,
            "high_risk_activation_rate": high_activation,
            "targeted_events": targeted,
        }

    rows = []
    for seed in config["policy_seeds"]:
        rows.append(
            {
                "seed": seed,
                "training_paths_equal": True,
                "training_dose_exact": True,
                "training_no_routing_dose_exact": True,
                "gate_off_exact": True,
                "convergence": {"stable": True},
                "final_evaluation": {
                    "paths_equal": True,
                    "dose_exact": True,
                    "no_routing_dose_exact": True,
                    "methods": {
                        "no_intervention": metrics(6.3, 10.0, 0.0, 0.0, 0),
                        "risk_aware": metrics(5.9, 9.8, 0.2, 0.04, 150),
                        "matched_random": metrics(6.0, 9.8, 0.2, 0.18, 150),
                        "spread_only_threshold": metrics(5.85, 9.5, 0.35, 0.25, 250),
                        "always_on": metrics(5.7, 8.0, 1.0, 1.0, 800),
                        "no_routing": metrics(6.2, 10.0, 0.2, 0.04, 0),
                    },
                },
            }
        )
    result = aggregate(rows, config)
    assert result["status"] == "pass"
    assert all(result["checks"].values())


def test_v4_fixed_horizon_population_gate_keeps_all_policies() -> None:
    config = load_config("abides_mappo_adaptive_v4_base_development_v1.json")
    history = [
        {
            "completed_episodes": episode,
            "mean_half_spread": spread,
            "mean_reward": reward,
        }
        for episode, spread, reward in (
            (336, 6.30, 10.00),
            (360, 6.34, 10.05),
            (384, 6.36, 10.08),
        )
    ]
    diagnostic = fixed_horizon_diagnostic(history, config)
    assert diagnostic["stable"]
    rows = [
        {
            "seed": seed,
            "status": "complete",
            "completed_episodes": 384,
            "fixed_horizon": diagnostic,
        }
        for seed in config["base_training_seeds"]
    ]
    summary = summarize_base_rows(rows, config)
    assert summary["status"] == "pass"
    assert all(summary["checks"].values())


def test_v4_population_gate_fails_when_more_than_two_spreads_drift() -> None:
    config = load_config("abides_mappo_adaptive_v4_base_development_v1.json")
    stable = fixed_horizon_diagnostic(
        [
            {"mean_half_spread": 6.30, "mean_reward": 10.0},
            {"mean_half_spread": 6.34, "mean_reward": 10.0},
            {"mean_half_spread": 6.36, "mean_reward": 10.0},
        ],
        config,
    )
    drifting = fixed_horizon_diagnostic(
        [
            {"mean_half_spread": 6.00, "mean_reward": 10.0},
            {"mean_half_spread": 6.15, "mean_reward": 10.0},
            {"mean_half_spread": 6.25, "mean_reward": 10.0},
        ],
        config,
    )
    rows = []
    for index, seed in enumerate(config["base_training_seeds"]):
        rows.append(
            {
                "seed": seed,
                "status": "complete",
                "completed_episodes": 384,
                "fixed_horizon": drifting if index < 3 else stable,
            }
        )
    summary = summarize_base_rows(rows, config)
    assert summary["status"] == "fail"
    assert not summary["checks"]["spread_stable_seed_rate"]
