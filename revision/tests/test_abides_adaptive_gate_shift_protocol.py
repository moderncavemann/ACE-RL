from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_adaptive_gate_diagnostic_uses_fresh_seeds_and_exogenous_features() -> None:
    config = json.loads(
        (
            ROOT / "revision/configs/abides_adaptive_gate_shift_diagnostic_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["risk_feature_names"] == [
        "public_volatility",
        "absolute_lagged_return",
    ]
    assert set(config["adaptive_training_seeds"]).isdisjoint(
        config["diagnostic_environment_seeds"]
    )
    assert set(config["diagnostic_environment_seeds"]).isdisjoint(range(420000, 420010))
    assert config["maximum_high_risk_activation_rate"] == 0.15
