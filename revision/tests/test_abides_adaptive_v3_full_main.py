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
    spread_only_decision,
)
from experiments.run_abides_mappo_adaptive_v3_response import (  # noqa: E402
    market_config,
)
from experiments.run_abides_mappo_full_main import protocol_fingerprint  # noqa: E402
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
