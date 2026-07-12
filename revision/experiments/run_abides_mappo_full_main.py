#!/usr/bin/env python3
"""Frozen 20-seed ABIDES/MAPPO full-main runner."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_learned_gate_screen import (  # noqa: E402
    circular_within_risk,
    exact_dose,
    load_trainer,
    paired_summary,
    shuffled_within_risk,
    summarize,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402
from src.interventions.abides_gate import (  # noqa: E402
    ABIDESGateActivationController,
    ABIDESLearnedGateModel,
)


def has_modal_undercutter(actions: Sequence[int]) -> bool:
    half_spreads = np.asarray(actions, dtype=float) + 1.0
    values, counts = np.unique(half_spreads, return_counts=True)
    modal = float(np.max(values[counts == np.max(counts)]))
    return bool(np.any(half_spreads <= modal - 1.0))


def has_excess_spread(actions: Sequence[int]) -> bool:
    half_spreads = np.asarray(actions, dtype=float) + 1.0
    return bool(np.mean(half_spreads) > 4.0)


def run_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    episode_steps = int(config["episode_steps"])
    trainer = load_trainer(Path(config["policy_checkpoint"]), "cpu")
    model = ABIDESLearnedGateModel.load(config["gate_artifact"])
    action_namespace = int(config["action_namespace"])
    max_rate = float(config["max_activation_rate"])

    with ABIDESDockerDealerEnv(
        {"episode_steps": episode_steps}
    ) as env:
        learned_controller = ABIDESGateActivationController(
            episode_steps, max_activation_rate=max_rate
        )

        def learned_decision(state, actions, step):
            return learned_controller.decide(
                bool(model.decide(state, actions)["active"])
            )

        learned_rollout = rollout_episode(
            env,
            trainer,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            gate_decision=learned_decision,
            action_namespace=action_namespace,
        )
        learned_schedule = [
            bool(row["gate_active"]) for row in learned_rollout["records"]
        ]
        risk_path = [
            str(row["info"]["last_step_risk"])
            for row in learned_rollout["records"]
        ]
        matched_schedule = shuffled_within_risk(
            learned_schedule, risk_path, seed ^ 0xA117
        )
        circular_schedule = circular_within_risk(
            learned_schedule, risk_path, seed ^ 0xC117
        )
        if not exact_dose(learned_schedule, matched_schedule, risk_path):
            raise RuntimeError("matched-random dose mismatch")
        if not exact_dose(learned_schedule, circular_schedule, risk_path):
            raise RuntimeError("circular-shuffled dose mismatch")

        def scheduled(schedule, routing_enabled=True):
            return rollout_episode(
                env,
                trainer,
                seed=seed,
                episode_steps=episode_steps,
                train=False,
                gate_schedule=schedule,
                routing_enabled=routing_enabled,
                action_namespace=action_namespace,
            )

        no_intervention = scheduled([False] * episode_steps)
        matched_random = scheduled(matched_schedule)
        circular_shuffled = scheduled(circular_schedule)
        no_routing = scheduled(learned_schedule, routing_enabled=False)
        always_on = scheduled([True] * episode_steps)

        threshold_controller = ABIDESGateActivationController(
            episode_steps, max_activation_rate=max_rate
        )

        def threshold_decision(state, actions, step):
            proposal = has_excess_spread(actions) and has_modal_undercutter(actions)
            return threshold_controller.decide(proposal)

        spread_only_threshold = rollout_episode(
            env,
            trainer,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            gate_decision=threshold_decision,
            action_namespace=action_namespace,
        )

        oracle_controller = ABIDESGateActivationController(
            episode_steps, max_activation_rate=max_rate
        )

        def oracle_decision(state, actions, step):
            proposal = (
                state["info"].get("diagnostic_current_risk") == "low"
                and has_excess_spread(actions)
                and has_modal_undercutter(actions)
            )
            return oracle_controller.decide(proposal)

        oracle_diagnostic = rollout_episode(
            env,
            trainer,
            seed=seed,
            episode_steps=episode_steps,
            train=False,
            gate_decision=oracle_decision,
            action_namespace=action_namespace,
        )

    arm_rollouts = {
        "no_intervention": no_intervention,
        "learned_gate": learned_rollout,
        "matched_random": matched_random,
        "circular_shuffled": circular_shuffled,
        "spread_only_threshold": spread_only_threshold,
        "always_on": always_on,
        "no_routing": no_routing,
        "oracle_diagnostic": oracle_diagnostic,
    }
    return {
        "seed": seed,
        "methods": {
            method: summarize(rollout) for method, rollout in arm_rollouts.items()
        },
        "learned_raw_proposal_rate": learned_controller.proposal_rate,
        "learned_schedule": learned_schedule,
        "risk_path": risk_path,
        "matched_schedule": matched_schedule,
        "circular_schedule": circular_schedule,
    }


def aggregate(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    required_seed_count: int,
) -> dict[str, Any]:
    metrics = (
        "low_execution_half_spread",
        "mean_reward",
        "mean_volume",
        "activation_rate",
        "high_risk_activation_rate",
        "targeted_events",
    )
    method_summary = {}
    for method in methods:
        method_summary[method] = {
            metric: float(mean(row["methods"][method][metric] for row in rows))
            for metric in metrics
        }

    learned = [
        row["methods"]["learned_gate"]["low_execution_half_spread"]
        for row in rows
    ]
    contrasts = {}
    for comparator in (
        "matched_random",
        "circular_shuffled",
        "no_intervention",
        "spread_only_threshold",
        "always_on",
        "oracle_diagnostic",
    ):
        delta = [
            row["methods"][comparator]["low_execution_half_spread"]
            - learned_value
            for row, learned_value in zip(rows, learned)
        ]
        contrasts[f"{comparator}_minus_learned_gate"] = paired_summary(delta)
    integrity = {
        "required_seed_count": len(rows) == required_seed_count,
        "activation_in_range": all(
            0.05 <= row["methods"]["learned_gate"]["activation_rate"] <= 0.35
            for row in rows
        ),
        "volume_conserved": all(
            abs(row["methods"][method]["mean_volume"] - 20.0) < 1e-9
            for row in rows
            for method in methods
        ),
        "no_routing_equals_no_intervention": all(
            abs(
                row["methods"]["no_routing"]["low_execution_half_spread"]
                - row["methods"]["no_intervention"]["low_execution_half_spread"]
            )
            < 1e-9
            and abs(
                row["methods"]["no_routing"]["mean_reward"]
                - row["methods"]["no_intervention"]["mean_reward"]
            )
            < 1e-9
            for row in rows
        ),
    }
    return {
        "method_summary": method_summary,
        "paired_contrasts": contrasts,
        "integrity": integrity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="revision/configs/abides_mappo_full_main_v1.json"
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seeds:
        config["seeds"] = [int(value) for value in args.seeds.split(",")]
    if args.workers is not None:
        config["workers"] = args.workers
    if args.episode_steps is not None:
        config["episode_steps"] = args.episode_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    seeds = [int(value) for value in config["seeds"]]
    if args.allow_partial:
        config["required_seed_count"] = len(seeds)
    workers = min(int(config["workers"]), len(seeds))
    rows = []
    if workers == 1:
        for seed in seeds:
            row = run_seed(seed, config)
            rows.append(row)
            print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_seed, seed, config): seed for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                row = future.result()
                rows.append(row)
                print(json.dumps({"seed": seed, "status": "complete"}), flush=True)
    rows.sort(key=lambda row: row["seed"])
    aggregated = aggregate(
        rows,
        config["methods"],
        int(config["required_seed_count"]),
    )
    summary = {
        "status": "complete"
        if all(aggregated["integrity"].values())
        else "integrity_fail",
        "protocol_version": config["protocol_version"],
        "backend": "abides",
        "seeds": seeds,
        "workers": workers,
        "episode_steps": config["episode_steps"],
        **aggregated,
    }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
