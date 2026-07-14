from __future__ import annotations

import json
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads(
        (
            REVISION_ROOT
            / "configs"
            / "abides_standard_learner_holdout_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_standard_learner_grid_is_frozen_before_validation() -> None:
    config = load_config()
    assert config["freeze_status"] == "frozen_before_hyperparameter_validation"
    assert config["formal_execution_allowed"] is False
    assert config["selected_hyperparameters"] is None
    assert config["selection_artifact_sha256"] is None
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
