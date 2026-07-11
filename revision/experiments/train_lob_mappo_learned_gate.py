#!/usr/bin/env python3
"""Train and freeze a risk-aware gate from passive local-LOB MAPPO states."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_mappo_contract_response_pilot import (  # noqa: E402
    calibrate_low_risk_threshold,
)
from run_lob_mappo_pilot import (  # noqa: E402
    agent_observations,
    choose_device,
    critic_observations,
    evaluation_episode_seed,
)
from run_lob_mappo_yoked_timing_pilot import (  # noqa: E402
    make_warm_trainer,
    train_episode,
)
from run_lob_marl_pilot import parse_seeds  # noqa: E402
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import (  # noqa: E402
    LocalLOBConfig,
    LocalLOBMarketEnv,
)
from src.interventions.learned_lob_gate import (  # noqa: E402
    FEATURE_NAMES,
    LearnedLOBGateModel,
    learned_gate_features,
    learned_gate_target,
)
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate  # noqa: E402


def make_environment_config(seed: int, episode_length: int) -> LocalLOBConfig:
    return LocalLOBConfig(
        n_agents=10,
        episode_length=episode_length,
        inventory_penalty=0.0001,
        base_flow_intensity=2.0,
        routing_mode="platform_reallocate",
        routing_routable_fraction=0.90,
        routing_platform_balanced=False,
        routing_contract="excess_spread_pool",
        routing_excess_reference_half_spread=4.0,
        routing_excess_scale_ticks=1.0,
        routing_relative_improvement_ticks=0.5,
        routing_target_kappa=4.0,
        routing_eligibility_mode="soft_rank",
        routing_target_half_spread_cap=3.0,
        risk_coupled_economics=True,
        mark_to_fundamental=True,
        informed_flow_prob=0.7,
        spread_elasticity=0.14,
        low_sigma=0.005,
        high_sigma=0.08,
        p_low_to_high=0.08,
        p_high_to_low=0.08,
        seed_liquidity_qty=0,
        agent_quote_quantity=1,
        quote_priority_mode="rotating",
        enforce_max_inventory=True,
        common_random_numbers=True,
        seed=seed,
    )


def train_base_policy(
    config: LocalLOBConfig,
    mappo: MAPPOConfig,
    seed: int,
    equilibrium_episodes: int,
    warm_steps: int,
    warm_updates: int,
) -> MAPPOTrainer:
    trainer, _ = make_warm_trainer(
        config,
        replace(mappo, policy_mode="multihead"),
        seed,
        warm_steps,
        warm_updates,
        3,
        6,
    )
    env = LocalLOBMarketEnv(config)
    no_gate = LOBInterventionGate(config, LOBGateConfig(), "none")
    for episode in range(equilibrium_episodes):
        train_episode(
            env,
            trainer,
            no_gate,
            0.0,
            seed + episode,
        )
    return trainer


def collect_gate_records(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    seed: int,
    episodes: int,
    expected_volume: float,
    namespace: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for episode in range(episodes):
        episode_seed = evaluation_episode_seed(
            seed,
            episode,
            namespace=namespace,
        )
        env = LocalLOBMarketEnv(config)
        env.reset(seed=episode_seed)
        done = False
        while not done:
            features = learned_gate_features(env, expected_volume)
            target = learned_gate_target(env)
            records.append({
                "seed": seed,
                "episode": episode,
                "step": env.t,
                "risk_state": str(env.risk_state),
                "target": int(target),
                "features": features.tolist(),
                "avg_agent_spread": float(np.mean(env.last_agent_spreads)),
            })
            obs = agent_observations(
                env,
                include_gate_signal=True,
                gate_signal=0.0,
            )
            critic_obs = critic_observations(env, obs)
            action_uniforms = np.random.default_rng(np.random.SeedSequence([
                int(episode_seed),
                int(env.t),
                71_117,
            ])).random(config.n_agents)
            actions, _, _ = trainer.act(
                obs,
                critic_obs,
                action_uniforms=action_uniforms,
            )
            _, _, done, _, _ = env.step(
                {i: int(actions[i]) for i in range(config.n_agents)},
                routing_flow_boost=0.0,
                gate_active=False,
            )
    return records


def prepare_records(
    seeds: Sequence[int],
    mappo: MAPPOConfig,
    equilibrium_episodes: int,
    calibration_episodes: int,
    calibration_quantile: float,
    calibration_upper_quantile: float,
    record_episodes: int,
    episode_length: int,
    warm_steps: int,
    warm_updates: int,
    expected_volume: float,
    namespace: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    all_records: List[Dict[str, object]] = []
    calibrations: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"gate-data seed={seed}", flush=True)
        config = make_environment_config(seed, episode_length)
        base = train_base_policy(
            config,
            mappo,
            seed,
            equilibrium_episodes,
            warm_steps,
            warm_updates,
        )
        calibration = calibrate_low_risk_threshold(
            config,
            base,
            seed,
            calibration_episodes,
            calibration_quantile,
            calibration_upper_quantile,
        )
        contract_config = replace(
            config,
            routing_excess_reference_half_spread=float(
                calibration["threshold_spread"]
            ),
            routing_excess_scale_ticks=float(calibration["contract_scale_ticks"]),
        )
        calibrations.append({"seed": seed, **calibration})
        all_records.extend(collect_gate_records(
            contract_config,
            base,
            seed,
            record_episodes,
            expected_volume,
            namespace,
        ))
    return all_records, calibrations


def arrays(records: Sequence[Dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([record["features"] for record in records], dtype=float)
    labels = np.asarray([record["target"] for record in records], dtype=int)
    low_risk = features[:, 0]
    return features, labels, low_risk


def classification_metrics(
    labels: np.ndarray,
    low_risk: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    active = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    high_risk = low_risk < 0.5
    true_positive = int(np.sum(active & positive))
    false_positive = int(np.sum(active & negative))
    precision = true_positive / max(int(np.sum(active)), 1)
    recall = true_positive / max(int(np.sum(positive)), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "activation_rate": float(np.mean(active)),
        "target_prevalence": float(np.mean(positive)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_fpr": float(false_positive / max(int(np.sum(negative)), 1)),
        "high_risk_fpr": float(np.mean(active[high_risk])) if np.any(high_risk) else 0.0,
    }


def choose_probability_threshold(
    labels: np.ndarray,
    low_risk: np.ndarray,
    probabilities: np.ndarray,
    min_activation: float,
    max_activation: float,
    max_high_risk_fpr: float,
) -> tuple[float, List[Dict[str, float]]]:
    table: List[Dict[str, float]] = []
    candidates: List[tuple[float, Dict[str, float]]] = []
    for threshold in np.linspace(0.01, 0.99, 981):
        metrics = classification_metrics(labels, low_risk, probabilities, float(threshold))
        row = {"threshold": float(threshold), **metrics}
        table.append(row)
        if (
            min_activation <= metrics["activation_rate"] <= max_activation
            and metrics["high_risk_fpr"] <= max_high_risk_fpr
        ):
            candidates.append((float(threshold), metrics))
    if not candidates:
        raise RuntimeError("no learned-gate threshold satisfies validation constraints")
    selected_threshold, _ = max(
        candidates,
        key=lambda item: (
            item[1]["f1"],
            -item[1]["high_risk_fpr"],
            -abs(item[0] - 0.5),
        ),
    )
    return selected_threshold, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", default="35-39")
    parser.add_argument("--validation-seeds", default="40-44")
    parser.add_argument("--equilibrium-episodes", type=int, default=100)
    parser.add_argument("--calibration-episodes", type=int, default=5)
    parser.add_argument("--calibration-quantile", type=float, default=0.75)
    parser.add_argument("--calibration-upper-quantile", type=float, default=0.90)
    parser.add_argument("--record-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--warm-start-steps", type=int, default=200)
    parser.add_argument("--warm-start-updates", type=int, default=20)
    parser.add_argument("--expected-volume", type=float, default=2.5)
    parser.add_argument("--min-activation", type=float, default=0.05)
    parser.add_argument("--max-activation", type=float, default=0.35)
    parser.add_argument("--max-high-risk-fpr", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/lob_mappo_learned_gate_v1",
    )
    parser.add_argument(
        "--frozen-artifact",
        default=None,
        help="optional repo-visible compact learned-gate artifact",
    )
    args = parser.parse_args()

    train_seeds = parse_seeds(args.train_seeds)
    validation_seeds = parse_seeds(args.validation_seeds)
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("gate train and validation seeds must be disjoint")
    device = choose_device(args.device)
    config = make_environment_config(train_seeds[0], args.episode_length)
    mappo = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=config.n_actions,
        n_agents=config.n_agents,
        policy_mode="multihead",
        hidden_dim=64,
        learning_rate=1e-3,
        entropy_coef=0.005,
        update_epochs=4,
        minibatch_size=512,
        device=device,
    )

    train_records, train_calibrations = prepare_records(
        train_seeds,
        mappo,
        args.equilibrium_episodes,
        args.calibration_episodes,
        args.calibration_quantile,
        args.calibration_upper_quantile,
        args.record_episodes,
        args.episode_length,
        args.warm_start_steps,
        args.warm_start_updates,
        args.expected_volume,
        namespace=12_000_000,
    )
    validation_records, validation_calibrations = prepare_records(
        validation_seeds,
        mappo,
        args.equilibrium_episodes,
        args.calibration_episodes,
        args.calibration_quantile,
        args.calibration_upper_quantile,
        args.record_episodes,
        args.episode_length,
        args.warm_start_steps,
        args.warm_start_updates,
        args.expected_volume,
        namespace=13_000_000,
    )
    train_x, train_y, train_low = arrays(train_records)
    validation_x, validation_y, validation_low = arrays(validation_records)
    if len(np.unique(train_y)) != 2:
        raise RuntimeError("gate training records must contain both target classes")

    feature_mean = np.mean(train_x, axis=0)
    feature_scale = np.std(train_x, axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    logistic = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=17,
        solver="lbfgs",
    )
    logistic.fit((train_x - feature_mean) / feature_scale, train_y)
    train_probabilities = logistic.predict_proba(
        (train_x - feature_mean) / feature_scale
    )[:, 1]
    validation_probabilities = logistic.predict_proba(
        (validation_x - feature_mean) / feature_scale
    )[:, 1]
    threshold, threshold_table = choose_probability_threshold(
        validation_y,
        validation_low,
        validation_probabilities,
        args.min_activation,
        args.max_activation,
        args.max_high_risk_fpr,
    )
    model = LearnedLOBGateModel(
        feature_names=FEATURE_NAMES,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        coefficients=tuple(float(value) for value in logistic.coef_[0]),
        intercept=float(logistic.intercept_[0]),
        probability_threshold=float(threshold),
        expected_volume=float(args.expected_volume),
    )
    output = {
        "metadata": {
            "stage": "passive_local_lob_mappo_learned_gate_training",
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "equilibrium_episodes": args.equilibrium_episodes,
            "calibration_episodes": args.calibration_episodes,
            "calibration_quantile": args.calibration_quantile,
            "calibration_upper_quantile": args.calibration_upper_quantile,
            "record_episodes": args.record_episodes,
            "episode_length": args.episode_length,
            "device": device,
            "sklearn_version": sklearn.__version__,
            "label_definition": "low_risk AND calibrated_contract_severity_gt_zero",
            "uses_intervention_outcomes": False,
            "environment_config": asdict(config),
            "mappo_config": asdict(mappo),
        },
        "model": model.to_dict(),
        "train_metrics": classification_metrics(
            train_y,
            train_low,
            train_probabilities,
            threshold,
        ),
        "validation_metrics": classification_metrics(
            validation_y,
            validation_low,
            validation_probabilities,
            threshold,
        ),
        "train_calibrations": train_calibrations,
        "validation_calibrations": validation_calibrations,
        "threshold_table": threshold_table,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "learned_gate.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    if args.frozen_artifact:
        frozen_path = Path(args.frozen_artifact)
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(
            json.dumps({
                "metadata": output["metadata"],
                "model": output["model"],
                "train_metrics": output["train_metrics"],
                "validation_metrics": output["validation_metrics"],
                "train_calibrations": output["train_calibrations"],
                "validation_calibrations": output["validation_calibrations"],
            }, indent=2),
            encoding="utf-8",
        )
    with (output_dir / "train_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record) + "\n")
    with (output_dir / "validation_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in validation_records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "frozen_artifact": args.frozen_artifact,
        "threshold": threshold,
        "train_metrics": output["train_metrics"],
        "validation_metrics": output["validation_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
