from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))

from src.interventions.abides_gate import (
    ABIDESGateActivationController,
    ABIDESTwoStageGateModel,
    abides_gate_features,
)


def sample_state(risk: str) -> dict:
    observations = [[0.0, 1.0, 0.9, 0.2, 0.1, 0.25] for _ in range(10)]
    return {
        "observations": observations,
        "info": {
            "flow_shares": [0.1] * 10,
            "public_volatility": 0.8,
            "diagnostic_current_risk": risk,
        },
    }


def test_activation_controller_never_exceeds_frozen_budget() -> None:
    controller = ABIDESGateActivationController(episode_steps=40)
    accepted = [controller.decide(True) for _ in range(40)]
    assert sum(accepted) == 14
    assert controller.activation_rate == 0.35
    assert controller.proposal_rate == 1.0


def test_learned_features_ignore_diagnostic_risk() -> None:
    actions = [5] + [6] * 9
    low = abides_gate_features(sample_state("low"), actions)
    high = abides_gate_features(sample_state("high"), actions)
    np.testing.assert_array_equal(low, high)


def test_two_stage_gate_separates_risk_probability_from_opportunity() -> None:
    model = ABIDESTwoStageGateModel.load(
        ROOT / "revision/configs/abides_two_stage_gate_v1.json"
    )
    state = sample_state("low")
    aligned = model.decide(state, [6] * 10)
    undercut = model.decide(state, [5] + [6] * 9)
    assert aligned["risk_probability"] == undercut["risk_probability"]
    assert aligned["opportunity"] is False
    assert undercut["opportunity"] is True
    assert aligned["active"] is False


def test_full_main_seeds_are_disjoint_from_all_gate_splits() -> None:
    config = json.loads(
        (ROOT / "revision/configs/abides_mappo_full_main_v1.json").read_text(
            encoding="utf-8"
        )
    )
    full = set(config["seeds"])
    prior = set(range(300000, 300005))
    prior.update(range(310000, 310005))
    prior.update(range(320000, 320005))
    prior.update(range(330000, 330005))
    assert len(full) == 20
    assert not full & prior
