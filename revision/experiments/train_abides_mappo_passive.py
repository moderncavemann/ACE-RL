#!/usr/bin/env python3
"""Train and freeze the no-intervention ABIDES/MAPPO assay policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.abides_mappo import make_abides_trainer, rollout_episode  # noqa: E402
from src.envs.abides_docker_env import ABIDESDockerDealerEnv  # noqa: E402


def action_histogram(rows: list[dict]) -> list[int]:
    values = [
        action
        for row in rows
        for record in row["records"]
        for action in record["actions"]
    ]
    return np.bincount(values, minlength=7).astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seed", type=int, default=900)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/abides_mappo_passive_v1",
    )
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    output_dir = ROOT.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = make_abides_trainer(
        seed=args.train_seed,
        episode_steps=args.episode_steps,
        device=args.device,
    )
    market_config = {"episode_steps": args.episode_steps}
    training = []
    started = time.perf_counter()
    with ABIDESDockerDealerEnv(market_config) as env:
        for episode in range(args.episodes):
            row = rollout_episode(
                env,
                trainer,
                seed=100_000 + args.train_seed * 1_000 + episode,
                episode_steps=args.episode_steps,
                train=True,
            )
            training.append(
                {
                    "episode": episode,
                    "mean_half_spread": row["mean_half_spread"],
                    "mean_reward": row["mean_reward"],
                    **row["update"],
                }
            )
            if (episode + 1) % 10 == 0:
                recent = training[-10:]
                print(
                    json.dumps(
                        {
                            "episode": episode + 1,
                            "mean_half_spread_10": float(
                                np.mean([item["mean_half_spread"] for item in recent])
                            ),
                            "mean_reward_10": float(
                                np.mean([item["mean_reward"] for item in recent])
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        evaluation = []
        eval_rows = []
        for episode in range(args.eval_episodes):
            row = rollout_episode(
                env,
                trainer,
                seed=200_000 + episode,
                episode_steps=args.episode_steps,
                train=False,
                deterministic=False,
                action_namespace=91_117,
            )
            eval_rows.append(row)
            evaluation.append(
                {
                    "episode": episode,
                    "seed": 200_000 + episode,
                    "mean_half_spread": row["mean_half_spread"],
                    "mean_reward": row["mean_reward"],
                }
            )

    positive = sum(row["mean_half_spread"] > 4.0 for row in evaluation)
    passed = positive >= 7
    elapsed = time.perf_counter() - started
    checkpoint = {
        "trainer_config": asdict(trainer.config),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in trainer.model.state_dict().items()
        },
        "train_seed": args.train_seed,
        "episodes": args.episodes,
        "episode_steps": args.episode_steps,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    summary = {
        "status": "pass" if passed else "fail",
        "backend": "abides",
        "device": args.device,
        "train_seed": args.train_seed,
        "train_episodes": args.episodes,
        "episode_steps": args.episode_steps,
        "eval_episodes": args.eval_episodes,
        "eval_mean_half_spread": float(
            np.mean([row["mean_half_spread"] for row in evaluation])
        ),
        "eval_high_spread_episodes": positive,
        "eval_action_histogram": action_histogram(eval_rows),
        "elapsed_seconds": elapsed,
        "evaluation": evaluation,
    }
    (output_dir / "training.json").write_text(
        json.dumps(training, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
