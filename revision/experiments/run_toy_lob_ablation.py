#!/usr/bin/env python3
"""Gate ablation runner for after observation and baseline checks pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_toy_lob_baselines import load_gate_config, run_gate_mode
from src.envs.toy_lob_pricing_env import ToyLOBConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-config", default=None, help="Optional JSON output from calibrate_toy_lob_gate.py.")
    parser.add_argument("--output", default="revision/experiments/outputs/ablation/summary.json")
    args = parser.parse_args()

    config = ToyLOBConfig(seed=args.seed, risk_mode="mixed")
    gate_config = load_gate_config(args.gate_config)
    summary = {
        "full_gate": run_gate_mode(config, "full_gate", "risk_aware", args.episodes, args.eval_episodes, args.seed, gate_config=gate_config),
        "spread_only_gate": run_gate_mode(config, "spread_only_gate", "spread_only", args.episodes, args.eval_episodes, args.seed + 1000, gate_config=gate_config),
        "no_risk_adjustment": run_gate_mode(config, "no_risk_adjustment", "no_risk_adjustment", args.episodes, args.eval_episodes, args.seed + 2000, gate_config=gate_config),
        "no_alignment_feature": run_gate_mode(config, "no_alignment_feature", "no_alignment", args.episodes, args.eval_episodes, args.seed + 3000, gate_config=gate_config),
        "no_demand_suppression_feature": run_gate_mode(config, "no_demand_suppression_feature", "no_demand", args.episodes, args.eval_episodes, args.seed + 4000, gate_config=gate_config),
        "no_routing_intervention": run_gate_mode(config, "no_routing_intervention", "risk_aware", args.episodes, args.eval_episodes, args.seed + 5000, routing_rho=0.0, gate_config=gate_config),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved ablation summary to {output}")


if __name__ == "__main__":
    main()
