#!/usr/bin/env python3
"""Gate ablation runner for after observation and baseline checks pass."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_toy_lob_baselines import aggregate_seed_results, load_gate_config, parse_seeds, run_gate_mode
from src.interventions.gates import GateConfig, GateMode
from src.envs.toy_lob_pricing_env import ToyLOBConfig


ABLATIONS: List[Tuple[str, GateMode, int, float]] = [
    ("full_gate", "risk_aware", 0, 0.75),
    ("spread_only_gate", "spread_only", 1000, 0.75),
    ("no_risk_adjustment", "no_risk_adjustment", 2000, 0.75),
    ("no_alignment_feature", "no_alignment", 3000, 0.75),
    ("no_demand_suppression_feature", "no_demand", 4000, 0.75),
    ("no_routing_intervention", "risk_aware", 5000, 0.0),
]


def _run_ablation_job(job: Tuple[str, GateMode, int, float, int, int, int, GateConfig | None]) -> Dict[str, object]:
    name, gate_mode, seed_offset, routing_rho, episodes, eval_episodes, base_seed, gate_config = job
    run_seed = base_seed + seed_offset
    config = ToyLOBConfig(seed=base_seed, risk_mode="mixed")
    result = run_gate_mode(
        config,
        name,
        gate_mode,
        episodes,
        eval_episodes,
        run_seed,
        routing_rho=routing_rho,
        gate_config=gate_config,
    )
    result["base_seed"] = base_seed
    result["run_seed"] = run_seed
    return result


def run_ablation_grid(
    name: str,
    gate_mode: GateMode,
    seed_offset: int,
    routing_rho: float,
    episodes: int,
    eval_episodes: int,
    seeds: List[int],
    gate_config: GateConfig | None,
    jobs: int,
) -> List[Dict[str, object]]:
    ablation_jobs = [
        (name, gate_mode, seed_offset, routing_rho, episodes, eval_episodes, seed, gate_config)
        for seed in seeds
    ]
    if jobs <= 1 or len(ablation_jobs) <= 1:
        return [_run_ablation_job(job) for job in ablation_jobs]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_run_ablation_job, ablation_jobs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seeds. Overrides --seed.")
    parser.add_argument("--gate-config", default=None, help="Optional JSON output from calibrate_toy_lob_gate.py.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes across seeds.")
    parser.add_argument("--output", default="revision/experiments/outputs/ablation/summary.json")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds) if args.seeds else [args.seed]
    gate_config = load_gate_config(args.gate_config)
    summary: Dict[str, object] = {
        "metadata": {
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "seeds": seeds,
            "jobs": args.jobs,
            "gate_config_path": args.gate_config,
            "gate_config": vars(gate_config) if gate_config is not None else None,
        },
        "methods": {},
    }
    methods_out: Dict[str, object] = {}
    summary["methods"] = methods_out
    for name, gate_mode, seed_offset, routing_rho in ABLATIONS:
        results = run_ablation_grid(
            name,
            gate_mode,
            seed_offset,
            routing_rho,
            args.episodes,
            args.eval_episodes,
            seeds,
            gate_config,
            args.jobs,
        )
        methods_out[name] = {
            "seed_results": results,
            "aggregate": aggregate_seed_results(results),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved ablation summary to {output}")


if __name__ == "__main__":
    main()
