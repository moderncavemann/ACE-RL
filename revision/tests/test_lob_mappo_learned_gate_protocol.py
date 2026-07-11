"""Tests for the learned-gate matched-random timing protocol."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

REVISION_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REVISION_ROOT / "experiments"
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from run_lob_mappo_learned_gate_pilot import (  # noqa: E402
    make_environment_config,
    rates_from_counts,
    risk_dose_exact,
    risk_stratified_circular_schedule,
    risk_stratified_matched_schedule,
    schedule_counts,
    validate_gate_artifact,
)


class LearnedGateProtocolTest(unittest.TestCase):
    def test_matched_schedule_preserves_each_risk_stratum_dose(self) -> None:
        source = [True, False, True, False, True, False, False, True]
        risk = ["low", "low", "high", "high", "low", "high", "low", "high"]
        matched = risk_stratified_matched_schedule(source, risk, seed=11)
        self.assertTrue(risk_dose_exact(source, matched, risk))
        self.assertEqual(sum(source), sum(matched))

    def test_risk_dose_check_rejects_cross_stratum_shift(self) -> None:
        source = [True, False, False, False]
        matched = [False, False, True, False]
        risk = ["low", "low", "high", "high"]
        self.assertFalse(risk_dose_exact(source, matched, risk))

    def test_circular_schedule_preserves_stratum_dose_and_order(self) -> None:
        source = [True, True, False, False, False, True, False, True]
        risk = ["low", "low", "low", "low", "high", "high", "high", "high"]
        shifted = risk_stratified_circular_schedule(source, risk, seed=19)
        self.assertTrue(risk_dose_exact(source, shifted, risk))
        for regime in ("low", "high"):
            original = [value for value, state in zip(source, risk) if state == regime]
            candidate = [value for value, state in zip(shifted, risk) if state == regime]
            self.assertTrue(any(
                candidate == original[offset:] + original[:offset]
                for offset in range(len(original))
            ))

    def test_schedule_classification_rates(self) -> None:
        rows = [
            {"active": True, "target": True, "risk_state": "low"},
            {"active": False, "target": True, "risk_state": "low"},
            {"active": True, "target": False, "risk_state": "high"},
            {"active": False, "target": False, "risk_state": "high"},
        ]
        rates = rates_from_counts(schedule_counts(rows))
        self.assertEqual(rates["activation_rate"], 0.5)
        self.assertEqual(rates["precision"], 0.5)
        self.assertEqual(rates["recall"], 0.5)
        self.assertEqual(rates["high_risk_fpr"], 0.5)

    def artifact_payload(self) -> dict[str, object]:
        config = make_environment_config(seed=1, episode_length=100)
        return {
            "metadata": {
                "train_seeds": [1, 2],
                "validation_seeds": [3, 4],
                "uses_intervention_outcomes": False,
                "label_definition": (
                    "low_risk AND calibrated_contract_severity_gt_zero"
                ),
                "environment_config": asdict(config),
            },
            "validation_metrics": {
                "activation_rate": 0.10,
                "high_risk_fpr": 0.0,
            },
        }

    def test_artifact_provenance_accepts_disjoint_passive_data(self) -> None:
        config = make_environment_config(seed=5, episode_length=100)
        result = validate_gate_artifact(self.artifact_payload(), [5, 6], config)
        self.assertTrue(bool(result["disjoint"]))
        self.assertTrue(bool(result["passive_training_only"]))
        self.assertTrue(bool(result["environment_config_matches"]))

    def test_artifact_provenance_rejects_seed_overlap(self) -> None:
        config = make_environment_config(seed=4, episode_length=100)
        with self.assertRaises(ValueError):
            validate_gate_artifact(self.artifact_payload(), [4, 5], config)

    def test_artifact_provenance_rejects_intervention_outcomes(self) -> None:
        payload = self.artifact_payload()
        metadata = dict(payload["metadata"])
        metadata["uses_intervention_outcomes"] = True
        payload["metadata"] = metadata
        config = make_environment_config(seed=5, episode_length=100)
        with self.assertRaises(ValueError):
            validate_gate_artifact(payload, [5, 6], config)

    def test_artifact_provenance_rejects_environment_drift(self) -> None:
        config = make_environment_config(seed=5, episode_length=100)
        payload = self.artifact_payload()
        metadata = dict(payload["metadata"])
        environment = dict(metadata["environment_config"])
        environment["routing_target_kappa"] = 1.0
        metadata["environment_config"] = environment
        payload["metadata"] = metadata
        with self.assertRaises(ValueError):
            validate_gate_artifact(payload, [5, 6], config)


if __name__ == "__main__":
    unittest.main()
