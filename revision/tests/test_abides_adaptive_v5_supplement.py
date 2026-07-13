from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO = REVISION_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from experiments.run_abides_mappo_adaptive_v5_supplement import (  # noqa: E402
    EVALUATED_METHODS,
    NEW_METHODS,
    aggregate,
    circular_shift,
    circular_structure_exact,
)
from experiments.export_abides_mappo_adaptive_v5_supplement import (  # noqa: E402
    METHOD_ORDER,
    per_seed_rows,
)
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPublicRiskProxyModel,
)


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_circular_control_preserves_exact_signal_structure() -> None:
    source = [0.0, 0.7, 0.8, 0.0, 0.0, 0.9]
    shifted, offset = circular_shift(source, 530000)
    assert 0 < offset < len(source)
    assert sorted(shifted) == sorted(source)
    assert circular_structure_exact(source, shifted, offset)
    assert shifted != source


def test_public_proxy_is_single_feature_and_pre_action() -> None:
    artifact = load_config("abides_public_risk_proxies_v1/volatility_only.json")
    model = ABIDESPublicRiskProxyModel.from_dict(artifact)
    observations = np.zeros((10, 6), dtype=float)
    observations[:, 1] = 1.0
    state = {
        "observations": observations.tolist(),
        "info": {"public_volatility": 0.0, "flow_shares": [0.1] * 10},
    }
    calm = model.announced_strength(state)
    state["info"]["public_volatility"] = 4.0
    volatile = model.announced_strength(state)
    assert model.feature_name == "public_volatility"
    assert model.training_label == "simulator_low_risk"
    assert calm["probability"] > volatile["probability"]


def test_proxy_calibration_uses_passive_disjoint_records() -> None:
    summary = load_config("abides_public_risk_proxies_v1/calibration_summary.json")
    assert summary["training_label"] == "simulator_low_risk"
    assert summary["train_records"]["count"] == 1000
    assert summary["validation_records"]["count"] == 1000
    assert summary["train_records"]["sha256"] != summary["validation_records"]["sha256"]
    reference_rate = summary["reference_gate"]["paced_validation_metrics"][
        "activation_rate"
    ]
    for method in ("volatility_only", "return_only"):
        rate = summary["calibration"][method]["validation_metrics"][
            "activation_rate"
        ]
        assert abs(rate - reference_rate) <= 0.01


def test_supplement_smoke_config_keeps_v5_source_frozen() -> None:
    config = load_config("abides_mappo_adaptive_v5_supplement_smoke_v1.json")
    v5 = load_config("abides_mappo_adaptive_v5_full_main_primary_v1.json")
    assert tuple(config["new_methods"]) == NEW_METHODS
    assert config["v5_response_episodes"] == v5["response_episodes"] == 96
    assert config["response_environment_seed_namespace"] == v5[
        "response_environment_seed_namespace"
    ]
    assert config["response_action_namespace"] == v5["response_action_namespace"]
    assert set(config["evaluation_seeds"]).isdisjoint(v5["evaluation_seeds"])


def test_formal_supplement_is_frozen_on_all_v5_policies_and_fresh_paths() -> None:
    config = load_config("abides_mappo_adaptive_v5_supplement_v1.json")
    v5 = load_config("abides_mappo_adaptive_v5_full_main_primary_v1.json")
    assert config["policy_seeds"] == v5["policy_seeds"]
    assert config["response_episodes"] == v5["response_episodes"] == 96
    assert config["response_environment_seed_namespace"] == v5[
        "response_environment_seed_namespace"
    ]
    assert config["response_action_namespace"] == v5["response_action_namespace"]
    assert len(config["evaluation_seeds"]) == 10
    assert set(config["evaluation_seeds"]).isdisjoint(v5["evaluation_seeds"])
    assert config["required_policy_seed_count"] == 20
    assert config["minimum_positive_seed_rate"] == 0.70
    assert config["proxy_activation_tolerance"] == 0.05


def test_supplement_aggregate_accepts_identified_fixture() -> None:
    config = load_config("abides_mappo_adaptive_v5_supplement_smoke_v1.json")
    config["required_policy_seed_count"] = 2
    config["minimum_positive_seed_rate"] = 0.5

    def metrics(spread: float, activation: float, high_activation: float) -> dict:
        return {
            "low_execution_half_spread": spread,
            "high_execution_half_spread": 6.8,
            "mean_reward": 10.0,
            "dealer_reward_units_per_episode": 10_000.0,
            "mean_volume": 20.0,
            "activation_rate": activation,
            "high_risk_activation_rate": high_activation,
            "mean_signal_strength": activation * 0.8,
            "targeted_events_per_episode": activation * 1_000.0,
            "low_customer_cost_tick_units_per_episode": spread * 1_400.0,
            "low_risk_steps_per_episode": 70.0,
            "mean_spread_revenue": 1.0,
            "mean_adverse_selection_component": 0.2,
            "mean_inventory_penalty": 0.01,
            "mean_flow_hhi": 0.15,
            "mean_inventory_std": 2.0,
        }

    rows = []
    for seed in (530000, 530001):
        methods = {
            method: metrics(6.4, 0.20, 0.05) for method in EVALUATED_METHODS
        }
        methods["risk_aware"] = metrics(6.2, 0.20, 0.05)
        methods["circular_shifted"] = metrics(6.3, 0.20, 0.16)
        methods["matched_random"] = metrics(6.35, 0.20, 0.19)
        methods["no_intervention"] = metrics(6.5, 0.0, 0.0)
        methods["always_on"] = metrics(5.2, 1.0, 1.0)
        rows.append(
            {
                "seed": seed,
                "v5_model_digests": {method: method for method in (
                    "risk_aware",
                    "matched_random",
                    "spread_only_threshold",
                    "always_on",
                    "no_routing",
                )},
                "training_paths_equal": True,
                "training_circular_dose_exact": True,
                "training_circular_structure_exact": True,
                "gate_off_exact": True,
                "convergence": {"stable": True},
                "final_evaluation": {
                    "paths_equal": True,
                    "circular_dose_exact": True,
                    "circular_structure_exact": True,
                    "methods": methods,
                },
            }
        )
    result = aggregate(rows, config)
    assert result["status"] == "pass"
    assert result["checks"]["circular_timing_positive_ci"]
    assert result["paired_contrasts"]["circular_shifted_minus_risk_aware"][
        "positive"
    ] == 2


def test_paper_export_is_seed_sorted_and_recomputes_paired_deltas() -> None:
    def methods(risk_spread: float) -> dict:
        values = {}
        for offset, method in enumerate(METHOD_ORDER):
            values[method] = {
                "low_execution_half_spread": risk_spread + 0.01 * offset,
                "mean_reward": 10.0 - 0.01 * offset,
                "activation_rate": 0.2,
                "high_risk_activation_rate": 0.05,
            }
        values["risk_aware"]["low_execution_half_spread"] = risk_spread
        return values

    exported = per_seed_rows(
        [
            {"seed": 2, "final_evaluation": {"methods": methods(6.2)}},
            {"seed": 1, "final_evaluation": {"methods": methods(6.1)}},
        ]
    )
    assert [row["policy_seed"] for row in exported] == [1, 2]
    expected = (
        exported[0]["matched_random__low_execution_half_spread"]
        - exported[0]["risk_aware__low_execution_half_spread"]
    )
    assert exported[0]["matched_random__minus_risk_aware_spread"] == expected
