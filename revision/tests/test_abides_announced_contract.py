from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.interventions.abides_gate import (  # noqa: E402
    ABIDESPacedSignalController,
    ABIDESTwoStageGateModel,
)


ROOT = Path(__file__).resolve().parents[2]


def test_paced_controller_spreads_dose_across_the_episode() -> None:
    controller = ABIDESPacedSignalController(100, target_rate=0.35)
    schedule = [controller.decide(1.0) for _ in range(100)]
    assert sum(value > 0.0 for value in schedule) == 35
    assert sum(value > 0.0 for value in schedule[:50]) <= 18
    assert sum(value > 0.0 for value in schedule[50:]) >= 17


def test_two_stage_announced_signal_uses_lagged_quotes() -> None:
    artifact = json.loads(
        (ROOT / "revision/configs/abides_two_stage_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    model = ABIDESTwoStageGateModel.from_dict(artifact)
    observations = np.zeros((10, 6), dtype=float)
    observations[:, 1] = 1.0
    state = {
        "observations": observations.tolist(),
        "info": {
            "public_volatility": 0.0,
            "flow_shares": [0.1] * 10,
        },
    }
    decision = model.announced_strength(state)
    assert decision["proposed"]
    assert decision["severity"] == 1.0
    assert 0.0 < decision["strength"] <= 1.0

    observations[:, 1] = 0.0
    state["observations"] = observations.tolist()
    competitive = model.announced_strength(state)
    assert competitive["severity"] == 0.0
    assert competitive["strength"] == 0.0
