from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from run_abides_standard_learner_holdout import (  # noqa: E402
    stage_manifest_path,
)


def load_config() -> dict:
    return json.loads(
        (
            REVISION_ROOT
            / "configs"
            / "abides_standard_learner_holdout_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_standard_learner_grid_and_selection_are_frozen_before_formal() -> None:
    config = load_config()
    assert config["freeze_status"] == "frozen_after_validation_selection"
    assert config["formal_execution_allowed"] is True
    assert config["selected_hyperparameters"] == {
        "cell_id": "lr5e-4_entropy001",
        "learning_rate": 0.0005,
        "critic_learning_rate": 0.0005,
        "entropy_coefficient": 0.01,
    }
    assert config["selection_artifact_sha256"] == (
        "87979b0bfe0064a04ae9e69d402d38088daa22c121af17df11d7a113dd610dbc"
    )
    cells = config["hyperparameter_search_space"]
    assert len(cells) == 4
    assert {cell["learning_rate"] for cell in cells} == {0.0003, 0.0005}
    assert {cell["entropy_coefficient"] for cell in cells} == {0.01, 0.02}
    assert config["validation_episodes"] == 96
    assert config["official_algorithm_parameters"]["ppo_epoch"] == 10
    assert config["official_algorithm_parameters"]["recurrent_policy"] is True


def test_standard_learner_splits_and_modes_are_disjoint() -> None:
    config = load_config()
    roles = [
        set(config["policy_seeds"]),
        set(config["base_evaluation_seeds"]),
        set(config["response_evaluation_seeds"]),
        set(config["validation_seeds"]),
        set(config["validation_evaluation_seeds"]),
    ]
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            assert left.isdisjoint(right)
    assert config["learners"] == ["official_full_policy_MAPPO", "official_IPPO"]
    assert config["required_learner"] == "official_full_policy_MAPPO"
    assert config["base_episodes"] == 384
    assert config["response_episodes"] == 96
    assert config["primary_contrast"] == "matched_random_minus_risk_aware"


def test_standard_learner_selection_cannot_read_intervention_outcomes() -> None:
    rule = load_config()["hyperparameter_selection_rule"]
    assert "no_intervention_dealer_reward" in rule
    assert "risk_aware" not in rule
    assert "matched_random" not in rule


def test_standard_learner_stage_manifests_are_isolated(tmp_path: Path) -> None:
    validation = stage_manifest_path(tmp_path, "validation")
    formal = stage_manifest_path(tmp_path, "formal")
    assert validation == tmp_path / "validation" / "manifest.json"
    assert formal == tmp_path / "formal" / "manifest.json"
    assert validation != formal
    with pytest.raises(ValueError, match="unknown official learner stage"):
        stage_manifest_path(tmp_path, "unknown")
