from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO = REVISION_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from experiments.profile_abides_abce_mechanism_v1 import (  # noqa: E402
    economic_effects,
    hard_best_weights,
    parameter_counts,
    tie_aware_auc,
)
from experiments.verify_abides_abce_mechanism_profile_v1 import verify  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_profile_config_is_frozen_and_non_outcome() -> None:
    config = load_json(
        REVISION_ROOT / "configs" / "abides_abce_mechanism_profile_v1.json"
    )
    assert config["status"] == "frozen_non_outcome_profile"
    assert config["formal_outcome_experiment"] is False
    assert config["n_agents"] == 10
    assert config["warmup_iterations"] == 10_000
    assert config["measured_iterations"] == 100_000
    assert set(config["thread_limits"].values()) == {"1"}


def test_compact_response_parameter_counts_match_architecture() -> None:
    config = load_json(
        REVISION_ROOT / "configs" / "abides_abce_mechanism_profile_v1.json"
    )
    counts = parameter_counts(config["model_dimensions"])
    assert counts["base_actor_total"] == 9_222
    assert counts["centralized_critic"] == 5_697
    assert counts["base_actor_critic_total"] == 14_919
    assert counts["actor_adapter"] == 560
    assert counts["adapter_actor_critic_total"] == 15_479
    assert counts["response_trainable_including_critic"] == 6_257


def test_tie_aware_auc_uses_average_ranks() -> None:
    assert tie_aware_auc([0, 1, 0, 1], [0.0, 0.5, 0.5, 1.0]) == 0.875


def test_hard_best_allocation_is_demand_conserving() -> None:
    weights = hard_best_weights([3, 2, 2, 4], 0.8, 0.5)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[1] == weights[2]
    assert weights[1] > weights[0]


def test_economic_export_matches_frozen_primary_statistics() -> None:
    statistics = load_json(
        REVISION_ROOT
        / "paper"
        / "artifacts"
        / "abides_abce_primary_followup_v1"
        / "statistics.json"
    )
    effects = economic_effects(statistics)
    assert np.isclose(
        effects["low_risk_execution_improvement_ticks"], 0.03995670931651611
    )
    assert np.isclose(effects["low_risk_customer_tick_units_saved_per_episode"], 47.92)
    assert effects["all_state_direction"] == "aligned_is_worse"


def test_committed_mechanism_profile_bundle_verifies() -> None:
    config = REVISION_ROOT / "configs" / "abides_abce_mechanism_profile_v1.json"
    artifact = (
        REVISION_ROOT / "paper" / "artifacts" / "abides_abce_mechanism_profile_v1"
    )
    assert all(verify(config, artifact).values())
