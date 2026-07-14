from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from calibrate_abides_dose_routing_sensitivity import (  # noqa: E402
    budget_metrics,
    risk_candidate,
    spread_candidate,
)


def _episodes() -> list[list[dict]]:
    return [
        [
            {
                "policy_seed": 1,
                "evaluation_seed": 2,
                "step": step,
                "actions": [0, 1, 2, 3],
                "risk_probability": 0.9 if step % 2 == 0 else 0.2,
                "spread_severity": 0.8,
                "risk_state": "low" if step < 3 else "high",
            }
            for step in range(4)
        ]
    ]


def test_risk_candidate_applies_cutoff_before_pacing() -> None:
    result = risk_candidate(_episodes(), 1.0, 0.5, 4.0, 0.5)
    assert result["activation_rate"] == 0.5
    assert result["high_risk_activation_rate"] == 0.0


def test_spread_candidate_reports_budget_metrics() -> None:
    result = spread_candidate(_episodes(), 0.5, 0.0, 0.75, 4.0, 0.5)
    assert result["activation_rate"] == 0.5
    assert 0.0 < result["mean_routing_total_variation"] < 1.0


def test_budget_metrics_rejects_misaligned_records() -> None:
    try:
        budget_metrics(_episodes(), [[0.2]], 4.0, 0.5)
    except RuntimeError as exc:
        assert "counts differ" in str(exc)
    else:
        raise AssertionError("misaligned calibration records were accepted")
