from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_adaptive_pilot_has_disjoint_independent_seed_splits() -> None:
    config = json.loads(
        (ROOT / "revision/configs/abides_mappo_adaptive_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )
    training = set(config["training_seeds"])
    evaluation = set(config["evaluation_seeds"])
    prior = set(range(300000, 300005))
    prior.update(range(310000, 310005))
    prior.update(range(320000, 320005))
    prior.update(range(330000, 330005))
    prior.update(range(340000, 340020))
    assert len(training) == config["required_training_seed_count"] == 5
    assert len(evaluation) == 10
    assert training.isdisjoint(evaluation)
    assert training.isdisjoint(prior)
    assert evaluation.isdisjoint(prior)


def test_adaptive_pilot_core_methods_are_frozen() -> None:
    config = json.loads(
        (ROOT / "revision/configs/abides_mappo_adaptive_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["methods"] == [
        "no_intervention",
        "learned_gate",
        "matched_random",
        "spread_only_threshold",
        "always_on",
    ]
    assert config["checkpoint_interval_episodes"] == 1
    assert config["device"] == "cpu"


def test_two_stage_adaptive_pilot_uses_fresh_splits() -> None:
    config = json.loads(
        (
            ROOT / "revision/configs/abides_mappo_adaptive_two_stage_pilot_v1.json"
        ).read_text(encoding="utf-8")
    )
    prior = set(range(410000, 410005))
    prior.update(range(420000, 420010))
    prior.update(range(421000, 421005))
    assert config["gate_type"] == "two_stage"
    assert set(config["training_seeds"]).isdisjoint(prior)
    assert set(config["evaluation_seeds"]).isdisjoint(prior)
    assert set(config["training_seeds"]).isdisjoint(config["evaluation_seeds"])


def test_two_stage_extension_preserves_training_and_refreshes_evaluation() -> None:
    pilot = json.loads(
        (
            ROOT / "revision/configs/abides_mappo_adaptive_two_stage_pilot_v1.json"
        ).read_text(encoding="utf-8")
    )
    extension = json.loads(
        (
            ROOT / "revision/configs/abides_mappo_adaptive_two_stage_extension_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert extension["training_seeds"] == pilot["training_seeds"]
    assert extension["parent_completed_episodes"] == pilot["train_episodes"]
    assert extension["train_episodes"] == 150
    assert set(extension["evaluation_seeds"]).isdisjoint(pilot["evaluation_seeds"])
    assert extension["gate_artifact_sha256"] == pilot["gate_artifact_sha256"]
