"""Regression tests for the state-contingent platform routing contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.envs.local_lob_market_env import (  # noqa: E402
    LocalLOBConfig,
    LocalLOBMarketEnv,
)


class StateContingentRoutingContractTest(unittest.TestCase):
    def make_env(self) -> LocalLOBMarketEnv:
        config = LocalLOBConfig(
            n_agents=3,
            episode_length=2,
            routing_mode="platform_reallocate",
            routing_routable_fraction=0.9,
            routing_platform_balanced=False,
            routing_contract="excess_spread_pool",
            routing_excess_reference_half_spread=4.0,
            routing_excess_scale_ticks=1.0,
            routing_eligibility_mode="soft_rank",
            routing_target_kappa=4.0,
            seed_liquidity_qty=0,
            agent_quote_quantity=1,
            seed=3,
        )
        env = LocalLOBMarketEnv(config)
        env.reset(seed=3)
        return env

    def test_severity_is_zero_at_reference_and_one_after_scale(self) -> None:
        env = self.make_env()
        env.last_agent_spreads = np.full(3, 4.0)
        self.assertEqual(env.current_routing_contract_severity(), 0.0)
        env.last_agent_spreads = np.full(3, 4.5)
        self.assertEqual(env.current_routing_contract_severity(), 0.5)
        env.last_agent_spreads = np.full(3, 5.0)
        self.assertEqual(env.current_routing_contract_severity(), 1.0)

    def test_soft_rank_is_uniform_at_tie_and_rewards_relative_undercut(self) -> None:
        env = self.make_env()
        env.last_agent_spreads = np.full(3, 5.0)
        tied = env._target_agent_probabilities([0, 1, 2])
        np.testing.assert_allclose(tied, np.full(3, 1.0 / 3.0))

        env.last_agent_spreads = np.array([4.0, 5.0, 5.0])
        undercut = env._target_agent_probabilities([0, 1, 2])
        self.assertGreater(float(undercut[0]), float(tied[0]))
        self.assertGreater(float(undercut[0]), float(undercut[1]))

    def test_effective_alpha_tracks_announced_state_severity(self) -> None:
        env = self.make_env()
        actions = np.full(3, 4, dtype=np.int64)
        _, _, _, _, normal_info = env.step(
            actions,
            routing_flow_boost=1.0,
            routing_state_severity=0.0,
        )
        self.assertEqual(float(normal_info["routing_effective_alpha"]), 0.0)

        env.reset(seed=3)
        _, _, _, _, high_info = env.step(
            actions,
            routing_flow_boost=1.0,
            routing_state_severity=1.0,
        )
        self.assertEqual(float(high_info["routing_effective_alpha"]), 1.0)


if __name__ == "__main__":
    unittest.main()
