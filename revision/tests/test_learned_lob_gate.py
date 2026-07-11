"""Tests for the serializable passive-state learned LOB gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.learned_lob_gate import (  # noqa: E402
    FEATURE_NAMES,
    LearnedLOBGateModel,
    learned_gate_features,
    learned_gate_target,
)


class LearnedLOBGateTest(unittest.TestCase):
    def make_env(self) -> LocalLOBMarketEnv:
        env = LocalLOBMarketEnv(LocalLOBConfig(
            n_agents=3,
            routing_contract="excess_spread_pool",
            routing_excess_reference_half_spread=4.0,
            routing_excess_scale_ticks=1.0,
            seed=5,
        ))
        env.reset(seed=5)
        return env

    def test_target_requires_low_risk_and_positive_contract_severity(self) -> None:
        env = self.make_env()
        env.risk_state = "low"
        env.last_agent_spreads = np.full(3, 5.0)
        self.assertTrue(learned_gate_target(env))
        env.risk_state = "high"
        self.assertFalse(learned_gate_target(env))
        env.risk_state = "low"
        env.last_agent_spreads = np.full(3, 4.0)
        self.assertFalse(learned_gate_target(env))

    def test_feature_schema_contains_risk_severity_interaction(self) -> None:
        env = self.make_env()
        env.risk_state = "low"
        env.last_agent_spreads = np.full(3, 4.5)
        features = learned_gate_features(env, expected_volume=2.5)
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertEqual(features[0], 1.0)
        self.assertEqual(features[1], 0.5)
        self.assertEqual(features[5], 0.5)

    def test_model_round_trip_and_probability(self) -> None:
        coefficients = np.zeros(len(FEATURE_NAMES), dtype=float)
        coefficients[5] = 4.0
        model = LearnedLOBGateModel(
            feature_names=FEATURE_NAMES,
            feature_mean=tuple(0.0 for _ in FEATURE_NAMES),
            feature_scale=tuple(1.0 for _ in FEATURE_NAMES),
            coefficients=tuple(float(value) for value in coefficients),
            intercept=-1.0,
            probability_threshold=0.5,
            expected_volume=2.5,
        )
        active_features = np.zeros(len(FEATURE_NAMES), dtype=float)
        active_features[5] = 1.0
        inactive_features = np.zeros(len(FEATURE_NAMES), dtype=float)
        self.assertTrue(bool(model.decide(active_features)["active"]))
        self.assertFalse(bool(model.decide(inactive_features)["active"]))
        restored = LearnedLOBGateModel.from_dict({"model": model.to_dict()})
        self.assertAlmostEqual(
            restored.probability(active_features),
            model.probability(active_features),
        )


if __name__ == "__main__":
    unittest.main()
