#!/usr/bin/env python3
"""Two-stage MAPPO pilot for a state-contingent routing contract.

Stage one learns an unassisted market equilibrium. Stage two freezes that base
actor and trains a zero-initialized response adapter only when a low-risk
intervention is announced. The same frozen response policy is then evaluated
under oracle and dose-yoked timing, isolating timing from policy retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_mappo_frozen_policy_timing_pilot import (  # noqa: E402
    counterfactual_signal_response,
    fixed_rate_low_risk_schedule,
    risk_stratified_yoked_schedule,
)
from run_lob_mappo_pilot import choose_device, evaluation_episode_seed  # noqa: E402
from run_lob_mappo_yoked_timing_pilot import (  # noqa: E402
    episode_risk_path,
    evaluate_scheduled_policy,
    make_trainer,
    make_warm_trainer,
    mean_se,
    reference_schedule,
    train_episode,
)
from run_lob_marl_pilot import (  # noqa: E402
    load_lob_gate_config,
    parse_seeds,
    summarize_by_risk,
    summarize_records,
)
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate  # noqa: E402


def make_response_adapter(
    config: LocalLOBConfig,
    mappo: MAPPOConfig,
    base: MAPPOTrainer,
    seed: int,
) -> MAPPOTrainer:
    adapter_config = replace(
        mappo,
        policy_mode="intervention_adapter",
        policy_advantage_mode="relative_immediate",
    )
    trainer = make_trainer(config, adapter_config, seed)
    trainer.initialize_intervention_adapter(base)
    return trainer


def mean_training(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = (
        "entropy",
        "return_mean",
        "gate_activation_rate",
        "active_low_share",
        "routing_targeted_volume",
        "routing_contract_alpha",
        "routing_effective_alpha",
    )
    return {
        key: float(np.mean([row[key] for row in rows])) if rows else 0.0
        for key in keys
    }


def calibrate_low_risk_threshold(
    config: LocalLOBConfig,
    base: MAPPOTrainer,
    seed: int,
    episodes: int,
    quantile: float,
    upper_quantile: float,
) -> Dict[str, float]:
    records: List[Dict[str, object]] = []
    for episode in range(episodes):
        calibration_seed = evaluation_episode_seed(
            seed,
            episode,
            namespace=10_000_000,
        )
        records.extend(evaluate_scheduled_policy(
            config,
            base,
            [False] * config.episode_length,
            0.0,
            calibration_seed,
            "calibration_no_intervention",
            stochastic_actions=True,
            action_namespace=17_041,
        ))
    low_spreads = np.asarray([
        float(row["avg_agent_spread"])
        for row in records
        if str(row["risk_state"]) == "low"
    ], dtype=float)
    if len(low_spreads) == 0:
        raise RuntimeError("no low-risk records available for gate calibration")
    clipped_quantile = float(np.clip(quantile, 0.0, 1.0))
    clipped_upper_quantile = float(np.clip(upper_quantile, clipped_quantile, 1.0))
    threshold = float(np.quantile(low_spreads, clipped_quantile))
    upper = float(np.quantile(low_spreads, clipped_upper_quantile))
    return {
        "threshold_spread": threshold,
        "contract_scale_ticks": max(upper - threshold, 1.0 / config.n_agents),
        "quantile": clipped_quantile,
        "upper_quantile": clipped_upper_quantile,
        "n_low_risk_records": float(len(low_spreads)),
        "spread_mean": float(np.mean(low_spreads)),
        "spread_std": float(np.std(low_spreads)),
    }


def run_seed(
    config: LocalLOBConfig,
    mappo: MAPPOConfig,
    gate_config: LOBGateConfig,
    routing_rho: float,
    equilibrium_episodes: int,
    response_episodes: int,
    response_train_rate: float,
    calibration_episodes: int,
    calibration_quantile: float,
    calibration_upper_quantile: float,
    eval_episodes: int,
    seed: int,
    warm_steps: int,
    warm_updates: int,
    warm_low: int,
    warm_high: int,
) -> Dict[str, object]:
    base, warm = make_warm_trainer(
        config,
        replace(mappo, policy_mode="multihead"),
        seed,
        warm_steps,
        warm_updates,
        warm_low,
        warm_high,
    )
    no_gate = LOBInterventionGate(config, gate_config, "none")
    base_env = LocalLOBMarketEnv(config)
    equilibrium_training: List[Dict[str, float]] = []
    for episode in range(equilibrium_episodes):
        stats, _ = train_episode(
            base_env,
            base,
            no_gate,
            0.0,
            seed + episode,
        )
        equilibrium_training.append(stats)

    calibration = calibrate_low_risk_threshold(
        config,
        base,
        seed,
        calibration_episodes,
        calibration_quantile,
        calibration_upper_quantile,
    )
    calibrated_gate_config = replace(
        gate_config,
        threshold_spread=float(calibration["threshold_spread"]),
    )
    contract_config = replace(
        config,
        routing_excess_reference_half_spread=float(
            calibration["threshold_spread"]
        ),
        routing_excess_scale_ticks=float(calibration["contract_scale_ticks"]),
    )

    response = make_response_adapter(contract_config, mappo, base, seed)
    response_env = LocalLOBMarketEnv(contract_config)
    response_gate = LOBInterventionGate(contract_config, gate_config, "none")
    response_training: List[Dict[str, float]] = []
    training_schedule_exact: list[bool] = []
    for episode in range(response_episodes):
        episode_seed = seed + 100_000 + episode
        schedule = fixed_rate_low_risk_schedule(
            contract_config,
            episode_seed,
            response_train_rate,
            schedule_seed=seed * 3_000_017 + episode,
        )
        stats, replayed = train_episode(
            response_env,
            response,
            response_gate,
            routing_rho,
            episode_seed,
            active_schedule=schedule,
            announce_contract_strength=True,
        )
        response_training.append(stats)
        training_schedule_exact.append(schedule == replayed)

    records: Dict[str, List[Dict[str, object]]] = {
        "base_no_intervention": [],
        "response_timed": [],
        "response_yoked": [],
        "response_gate_off": [],
    }
    eval_dose_exact: list[bool] = []
    counterfactual_rows: list[Dict[str, float]] = []
    gate_off_equivalent: list[bool] = []
    for episode in range(eval_episodes):
        eval_seed = evaluation_episode_seed(seed, episode)
        timed_schedule = reference_schedule(
            contract_config,
            base,
            calibrated_gate_config,
            "oracle_low_high_spread_controlled",
            eval_seed,
            stochastic_actions=True,
            action_namespace=31_337,
        )
        yoked_schedule = risk_stratified_yoked_schedule(
            timed_schedule,
            episode_risk_path(contract_config, eval_seed),
            seed=seed * 4_000_037 + episode,
        )
        eval_dose_exact.append(sum(timed_schedule) == sum(yoked_schedule))
        base_rows = evaluate_scheduled_policy(
            contract_config,
            base,
            [False] * len(timed_schedule),
            0.0,
            eval_seed,
            "base_no_intervention",
            stochastic_actions=True,
            action_namespace=31_337,
            announce_contract_strength=True,
        )
        gate_off_rows = evaluate_scheduled_policy(
            contract_config,
            response,
            [False] * len(timed_schedule),
            0.0,
            eval_seed,
            "response_gate_off",
            stochastic_actions=True,
            action_namespace=31_337,
            announce_contract_strength=True,
        )
        gate_off_equivalent.append(all(
            str(base_row["risk_state"]) == str(response_row["risk_state"])
            and np.isclose(
                float(base_row["avg_agent_spread"]),
                float(response_row["avg_agent_spread"]),
                atol=0.0,
                rtol=0.0,
            )
            for base_row, response_row in zip(base_rows, gate_off_rows)
        ))
        records["base_no_intervention"].extend(base_rows)
        records["response_gate_off"].extend(gate_off_rows)
        records["response_timed"].extend(evaluate_scheduled_policy(
            contract_config,
            response,
            timed_schedule,
            routing_rho,
            eval_seed,
            "response_timed",
            stochastic_actions=True,
            action_namespace=31_337,
            announce_contract_strength=True,
        ))
        records["response_yoked"].extend(evaluate_scheduled_policy(
            contract_config,
            response,
            yoked_schedule,
            routing_rho,
            eval_seed,
            "response_yoked",
            stochastic_actions=True,
            action_namespace=31_337,
            announce_contract_strength=True,
        ))
        counterfactual_rows.append(counterfactual_signal_response(
            contract_config,
            response,
            eval_seed,
            threshold_spread=float(calibrated_gate_config.threshold_spread),
            stochastic_path=True,
            action_namespace=53_159,
        ))

    methods = {
        method: {
            "eval_all": summarize_records(rows),
            "eval_by_risk": summarize_by_risk(rows),
        }
        for method, rows in records.items()
    }
    methods["base_no_intervention"]["warm_start"] = warm
    methods["base_no_intervention"]["training"] = mean_training(equilibrium_training)
    methods["response_gate_off"]["training"] = mean_training(response_training)
    diagnostic_keys = tuple(counterfactual_rows[0]) if counterfactual_rows else ()
    return {
        "seed": seed,
        "methods": methods,
        "training_schedule_exact": bool(all(training_schedule_exact)),
        "eval_dose_exact": bool(all(eval_dose_exact)),
        "gate_off_base_equivalent": bool(all(gate_off_equivalent)),
        "calibration": calibration,
        "counterfactual_response_diagnostic": {
            key: float(np.mean([row[key] for row in counterfactual_rows]))
            for key in diagnostic_keys
        },
    }


def aggregate(
    seed_results: Sequence[Dict[str, object]],
    min_expected_response: float,
) -> Dict[str, object]:
    def risk_metric(result: Dict[str, object], method: str, risk: str, key: str) -> float:
        method_block = dict(dict(result["methods"])[method])
        return float(dict(dict(method_block["eval_by_risk"])[risk])[key])

    def all_metric(result: Dict[str, object], method: str, key: str) -> float:
        method_block = dict(dict(result["methods"])[method])
        return float(dict(method_block["eval_all"])[key])

    low_deltas = [
        risk_metric(result, "response_yoked", "low", "avg_agent_spread_mean")
        - risk_metric(result, "response_timed", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    high_deltas = [
        risk_metric(result, "response_yoked", "high", "avg_agent_spread_mean")
        - risk_metric(result, "response_timed", "high", "avg_agent_spread_mean")
        for result in seed_results
    ]
    no_low = [
        risk_metric(result, "base_no_intervention", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    no_minus_timed = [
        risk_metric(result, "base_no_intervention", "low", "avg_agent_spread_mean")
        - risk_metric(result, "response_timed", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    response_gaps = [
        float(dict(result["counterfactual_response_diagnostic"])[
            "low_risk_high_spread_expected_half_spread_reduction"
        ])
        for result in seed_results
    ]
    calibrated_thresholds = [
        float(dict(result["calibration"])["threshold_spread"])
        for result in seed_results
    ]
    activation = float(np.mean([
        all_metric(result, "response_timed", "gate_active_mean")
        for result in seed_results
    ]))
    timed_effective_alpha = float(np.mean([
        all_metric(result, "response_timed", "routing_effective_alpha_mean")
        for result in seed_results
    ]))
    yoked_effective_alpha = float(np.mean([
        all_metric(result, "response_yoked", "routing_effective_alpha_mean")
        for result in seed_results
    ]))
    timed_rewards = [
        all_metric(result, "response_timed", "mean_reward_mean")
        for result in seed_results
    ]
    yoked_rewards = [
        all_metric(result, "response_yoked", "mean_reward_mean")
        for result in seed_results
    ]
    checks = {
        "training_schedule_exact": bool(all(
            bool(result["training_schedule_exact"]) for result in seed_results
        )),
        "eval_dose_exact": bool(all(bool(result["eval_dose_exact"]) for result in seed_results)),
        "gate_off_adapter_exactly_matches_base": bool(all(
            bool(result["gate_off_base_equivalent"]) for result in seed_results
        )),
        "no_intervention_failure_rate_at_least_70pct": bool(
            np.mean(np.asarray(no_low) >= 4.0) >= 0.70
        ),
        "eval_activation_in_5_to_40pct": bool(0.05 <= activation <= 0.40),
        "timed_contract_allocates_more_effective_pool_than_yoked": bool(
            timed_effective_alpha > yoked_effective_alpha
        ),
        "counterfactual_expected_response_is_material": bool(
            float(np.mean(response_gaps)) >= min_expected_response
        ),
        "counterfactual_response_positive_in_at_least_70pct_seeds": bool(
            np.mean(np.asarray(response_gaps) > 0.0) >= 0.70
        ),
        "timed_beats_yoked_on_low_risk_spread": bool(float(np.mean(low_deltas)) > 0.0),
        "paired_positive_seed_rate_at_least_70pct": bool(
            np.mean(np.asarray(low_deltas) > 0.0) >= 0.70
        ),
        "timed_improves_base_low_risk_spread": bool(float(np.mean(no_minus_timed)) > 0.0),
        "high_risk_spread_not_reduced_more_than_half_tick": bool(
            float(np.mean(high_deltas)) <= 0.5
        ),
        "reward_not_collapsed_vs_yoked": bool(
            float(np.mean(timed_rewards)) >= float(np.mean(yoked_rewards)) - 0.005
        ),
    }
    return {
        "base_no_intervention_low_risk_spreads": no_low,
        "paired_yoked_minus_timed_low_risk_spread": {
            **mean_se(low_deltas),
            "positive_seed_count": int(np.sum(np.asarray(low_deltas) > 0.0)),
            "n_seeds": len(low_deltas),
            "values": low_deltas,
        },
        "paired_yoked_minus_timed_high_risk_spread": {
            **mean_se(high_deltas),
            "values": high_deltas,
        },
        "paired_base_minus_timed_low_risk_spread": {
            **mean_se(no_minus_timed),
            "values": no_minus_timed,
        },
        "counterfactual_signal_response": {
            **mean_se(response_gaps),
            "positive_seed_count": int(np.sum(np.asarray(response_gaps) > 0.0)),
            "n_seeds": len(response_gaps),
            "values": response_gaps,
        },
        "calibrated_threshold_spread": mean_se(calibrated_thresholds),
        "eval_activation_rate": activation,
        "timed_effective_alpha": timed_effective_alpha,
        "yoked_effective_alpha": yoked_effective_alpha,
        "pass_checks": checks,
        "passes_all": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equilibrium-episodes", type=int, default=100)
    parser.add_argument("--response-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--response-train-rate", type=float, default=1.00)
    parser.add_argument("--calibration-episodes", type=int, default=5)
    parser.add_argument("--calibration-quantile", type=float, default=0.75)
    parser.add_argument("--calibration-upper-quantile", type=float, default=0.90)
    parser.add_argument("--min-expected-response", type=float, default=0.05)
    parser.add_argument("--threshold-spread", type=float, default=4.0)
    parser.add_argument("--activation-cap", type=float, default=0.35)
    parser.add_argument("--routing-rho", type=float, default=1.0)
    parser.add_argument("--routing-routable-fraction", type=float, default=0.90)
    parser.add_argument("--routing-excess-reference-half-spread", type=float, default=4.0)
    parser.add_argument("--routing-excess-scale-ticks", type=float, default=1.0)
    parser.add_argument("--routing-target-kappa", type=float, default=4.0)
    parser.add_argument("--gate-config", default=None)
    parser.add_argument("--warm-start-steps", type=int, default=200)
    parser.add_argument("--warm-start-updates", type=int, default=20)
    parser.add_argument("--warm-start-low-half-spread", type=int, default=3)
    parser.add_argument("--warm-start-high-half-spread", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/lob_mappo_contract_response_pilot_v1",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    config = LocalLOBConfig(
        n_agents=10,
        episode_length=args.episode_length,
        inventory_penalty=0.0001,
        base_flow_intensity=2.0,
        routing_mode="platform_reallocate",
        routing_routable_fraction=args.routing_routable_fraction,
        routing_platform_balanced=False,
        routing_contract="excess_spread_pool",
        routing_excess_reference_half_spread=args.routing_excess_reference_half_spread,
        routing_excess_scale_ticks=args.routing_excess_scale_ticks,
        routing_relative_improvement_ticks=0.5,
        routing_target_kappa=args.routing_target_kappa,
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
        seed=seeds[0] if seeds else 0,
    )
    gate_config = load_lob_gate_config(args.gate_config)
    gate_config = LOBGateConfig(**{
        **vars(gate_config),
        "threshold_spread": args.threshold_spread,
        "activation_cap": args.activation_cap,
    })
    mappo = MAPPOConfig(
        obs_dim=1,
        critic_obs_dim=1,
        n_actions=config.n_actions,
        n_agents=config.n_agents,
        policy_mode="multihead",
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    seed_results: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"seed={seed} equilibrium+contract-response", flush=True)
        seed_results.append(run_seed(
            replace(config, seed=seed),
            mappo,
            gate_config,
            args.routing_rho,
            args.equilibrium_episodes,
            args.response_episodes,
            args.response_train_rate,
            args.calibration_episodes,
            args.calibration_quantile,
            args.calibration_upper_quantile,
            args.eval_episodes,
            seed,
            args.warm_start_steps,
            args.warm_start_updates,
            args.warm_start_low_half_spread,
            args.warm_start_high_half_spread,
        ))

    summary = {
        "metadata": {
            "stage": "local_lob_mappo_contract_response_pilot_not_full_main",
            "device": device,
            "seeds": seeds,
            "equilibrium_episodes": args.equilibrium_episodes,
            "response_episodes": args.response_episodes,
            "eval_episodes": args.eval_episodes,
            "response_train_rate": args.response_train_rate,
            "calibration_episodes": args.calibration_episodes,
            "calibration_quantile": args.calibration_quantile,
            "calibration_upper_quantile": args.calibration_upper_quantile,
            "min_expected_response": args.min_expected_response,
            "same_frozen_response_policy_for_timed_and_yoked_eval": True,
            "gate_off_schedule_derived_from_base_policy": True,
            "risk_stratified_exact_dose_yoking": True,
            "base_actor_frozen_during_response_training": True,
            "response_adapter_active_only_in_low_risk_announced_windows": True,
            "response_policy_advantage_mode": "relative_immediate",
            "config": asdict(config),
            "gate_config": asdict(gate_config),
            "mappo_config": asdict(mappo),
        },
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results, args.min_expected_response),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "paired_low_risk": summary["aggregate"][
            "paired_yoked_minus_timed_low_risk_spread"
        ],
        "counterfactual_response": summary["aggregate"][
            "counterfactual_signal_response"
        ],
        "pass_checks": summary["aggregate"]["pass_checks"],
        "passes_all": summary["aggregate"]["passes_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
