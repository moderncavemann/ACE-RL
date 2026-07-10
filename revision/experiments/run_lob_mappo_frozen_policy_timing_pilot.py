#!/usr/bin/env python3
"""Isolate timing with one frozen, intervention-responsive seller policy."""

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

from run_lob_mappo_pilot import (  # noqa: E402
    agent_observations,
    choose_device,
    critic_observations,
    evaluation_episode_seed,
)
from run_lob_mappo_yoked_timing_pilot import (  # noqa: E402
    evaluate_scheduled_policy,
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


def fixed_rate_schedule(length: int, rate: float, seed: int) -> list[bool]:
    active_count = int(round(float(np.clip(rate, 0.0, 1.0)) * length))
    schedule = np.zeros(length, dtype=bool)
    schedule[:active_count] = True
    np.random.default_rng(seed).shuffle(schedule)
    return schedule.tolist()


def episode_risk_path(config: LocalLOBConfig, seed: int) -> list[str]:
    """Return the pre-action risk state path for a CRN episode."""
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    actions = np.zeros(config.n_agents, dtype=np.int64)
    risk_path: list[str] = []
    done = False
    while not done:
        risk_path.append(str(env.risk_state))
        _, _, done, _, _ = env.step(actions)
    return risk_path


def fixed_rate_low_risk_schedule(
    config: LocalLOBConfig,
    episode_seed: int,
    low_risk_rate: float,
    schedule_seed: int,
) -> list[bool]:
    risk_path = episode_risk_path(config, episode_seed)
    low_indices = np.asarray(
        [index for index, risk in enumerate(risk_path) if risk == "low"],
        dtype=int,
    )
    active_count = int(round(
        float(np.clip(low_risk_rate, 0.0, 1.0)) * len(low_indices)
    ))
    rng = np.random.default_rng(schedule_seed)
    rng.shuffle(low_indices)
    schedule = np.zeros(len(risk_path), dtype=bool)
    schedule[low_indices[:active_count]] = True
    return schedule.tolist()


def risk_stratified_yoked_schedule(
    schedule: Sequence[bool],
    risk_path: Sequence[str],
    seed: int,
) -> list[bool]:
    if len(schedule) != len(risk_path):
        raise ValueError("schedule and risk path must have the same length")
    high_active = [
        index for index, (active, risk) in enumerate(zip(schedule, risk_path))
        if bool(active) and risk != "low"
    ]
    if high_active:
        raise ValueError("risk-stratified yoking requires a low-risk-only source schedule")
    low_indices = np.asarray(
        [index for index, risk in enumerate(risk_path) if risk == "low"],
        dtype=int,
    )
    low_values = np.asarray([bool(schedule[index]) for index in low_indices], dtype=bool)
    np.random.default_rng(seed).shuffle(low_values)
    yoked = np.zeros(len(schedule), dtype=bool)
    yoked[low_indices] = low_values
    return yoked.tolist()


def counterfactual_signal_response(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    seed: int,
    threshold_spread: float,
) -> Dict[str, float]:
    """Compare signal-off and signal-on actions at the exact same states.

    The environment follows the signal-off action so the state path is not
    affected by the counterfactual signal-on query. This isolates whether the
    frozen seller policy learned to use the announced intervention signal.
    """
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    all_reductions: list[float] = []
    low_reductions: list[float] = []
    low_high_spread_reductions: list[float] = []
    all_action_changes: list[float] = []
    low_cap_compliance_gains: list[float] = []
    spread_levels = np.asarray(config.half_spread_levels, dtype=float)
    cap = float(config.routing_target_half_spread_cap)
    done = False
    while not done:
        risk_state = str(env.risk_state)
        prior_avg_spread = float(np.mean(env.last_agent_spreads))
        obs_off = agent_observations(env, include_gate_signal=True, gate_signal=0.0)
        obs_on = agent_observations(env, include_gate_signal=True, gate_signal=1.0)
        critic_off = critic_observations(env, obs_off)
        critic_on = critic_observations(env, obs_on)
        actions_off, _, _ = trainer.act(obs_off, critic_off, deterministic=True)
        actions_on, _, _ = trainer.act(obs_on, critic_on, deterministic=True)
        spreads_off = spread_levels[actions_off]
        spreads_on = spread_levels[actions_on]
        reduction = float(np.mean(spreads_off - spreads_on))
        all_reductions.append(reduction)
        all_action_changes.append(float(np.mean(actions_off != actions_on)))
        if risk_state == "low":
            low_reductions.append(reduction)
            low_cap_compliance_gains.append(float(
                np.mean(spreads_on <= cap) - np.mean(spreads_off <= cap)
            ))
            if prior_avg_spread >= threshold_spread:
                low_high_spread_reductions.append(reduction)
        _, _, done, _, _ = env.step(
            {i: int(actions_off[i]) for i in range(config.n_agents)},
            routing_flow_boost=0.0,
            gate_active=False,
            gate_score=0.0,
        )

    def safe_mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "all_state_half_spread_reduction": safe_mean(all_reductions),
        "low_risk_half_spread_reduction": safe_mean(low_reductions),
        "low_risk_high_spread_half_spread_reduction": safe_mean(
            low_high_spread_reductions
        ),
        "action_change_rate": safe_mean(all_action_changes),
        "low_risk_cap_compliance_gain": safe_mean(low_cap_compliance_gains),
        "low_risk_state_count": float(len(low_reductions)),
        "low_risk_high_spread_state_count": float(len(low_high_spread_reductions)),
    }


def run_seed(
    config: LocalLOBConfig,
    mappo: MAPPOConfig,
    gate_config: LOBGateConfig,
    routing_rho: float,
    response_train_rate: float,
    episodes: int,
    eval_episodes: int,
    seed: int,
    warm_steps: int,
    warm_updates: int,
    warm_low: int,
    warm_high: int,
    response_train_low_risk_only: bool,
    risk_stratified_yoke: bool,
    schedule_from_response_policy: bool,
) -> Dict[str, object]:
    reference_trainer, reference_warm = make_warm_trainer(
        config, mappo, seed, warm_steps, warm_updates, warm_low, warm_high
    )
    response_trainer, response_warm = make_warm_trainer(
        config, mappo, seed, warm_steps, warm_updates, warm_low, warm_high
    )
    reference_env = LocalLOBMarketEnv(config)
    response_env = LocalLOBMarketEnv(config)
    no_gate = LOBInterventionGate(config, gate_config, "none")
    response_gate = LOBInterventionGate(config, gate_config, "none")
    reference_training: List[Dict[str, float]] = []
    response_training: List[Dict[str, float]] = []
    training_dose_exact: list[bool] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        reference_stats, _ = train_episode(
            reference_env,
            reference_trainer,
            no_gate,
            0.0,
            episode_seed,
        )
        if response_train_low_risk_only:
            schedule = fixed_rate_low_risk_schedule(
                config,
                episode_seed,
                response_train_rate,
                schedule_seed=seed * 3_000_017 + episode,
            )
        else:
            schedule = fixed_rate_schedule(
                config.episode_length,
                response_train_rate,
                seed=seed * 3_000_017 + episode,
            )
        response_stats, replayed = train_episode(
            response_env,
            response_trainer,
            response_gate,
            routing_rho,
            episode_seed,
            active_schedule=schedule,
        )
        training_dose_exact.append(sum(schedule) == sum(replayed))
        reference_training.append(reference_stats)
        response_training.append(response_stats)

    records: Dict[str, List[Dict[str, object]]] = {
        "reference_no_intervention": [],
        "response_timed": [],
        "response_yoked": [],
        "response_gate_off": [],
    }
    eval_dose_exact: list[bool] = []
    counterfactual_diagnostics: list[Dict[str, float]] = []
    schedule_trainer = response_trainer if schedule_from_response_policy else reference_trainer
    for episode in range(eval_episodes):
        eval_seed = evaluation_episode_seed(seed, episode)
        timed_schedule = reference_schedule(
            config,
            schedule_trainer,
            gate_config,
            "oracle_low_high_spread_controlled",
            eval_seed,
        )
        permutation_seed = seed * 4_000_037 + episode
        if risk_stratified_yoke:
            yoked_schedule = risk_stratified_yoked_schedule(
                timed_schedule,
                episode_risk_path(config, eval_seed),
                permutation_seed,
            )
        else:
            permutation_rng = np.random.default_rng(permutation_seed)
            yoked_schedule = np.asarray(timed_schedule, dtype=bool)[
                permutation_rng.permutation(len(timed_schedule))
            ].tolist()
        eval_dose_exact.append(sum(timed_schedule) == sum(yoked_schedule))
        records["reference_no_intervention"].extend(evaluate_scheduled_policy(
            config,
            reference_trainer,
            [False] * len(timed_schedule),
            0.0,
            eval_seed,
            "reference_no_intervention",
        ))
        records["response_timed"].extend(evaluate_scheduled_policy(
            config,
            response_trainer,
            timed_schedule,
            routing_rho,
            eval_seed,
            "response_timed",
        ))
        records["response_yoked"].extend(evaluate_scheduled_policy(
            config,
            response_trainer,
            yoked_schedule,
            routing_rho,
            eval_seed,
            "response_yoked",
        ))
        records["response_gate_off"].extend(evaluate_scheduled_policy(
            config,
            response_trainer,
            [False] * len(timed_schedule),
            0.0,
            eval_seed,
            "response_gate_off",
        ))
        counterfactual_diagnostics.append(counterfactual_signal_response(
            config,
            response_trainer,
            eval_seed,
            threshold_spread=float(gate_config.threshold_spread),
        ))

    methods: Dict[str, object] = {}
    for method, rows in records.items():
        methods[method] = {
            "eval_all": summarize_records(rows),
            "eval_by_risk": summarize_by_risk(rows),
        }
    methods["reference_no_intervention"]["warm_start"] = reference_warm  # type: ignore[index]
    methods["response_gate_off"]["warm_start"] = response_warm  # type: ignore[index]
    methods["reference_no_intervention"]["training"] = {  # type: ignore[index]
        key: float(np.mean([row[key] for row in reference_training]))
        for key in ("entropy", "return_mean")
    }
    methods["response_gate_off"]["training"] = {  # type: ignore[index]
        key: float(np.mean([row[key] for row in response_training]))
        for key in (
            "entropy",
            "return_mean",
            "gate_activation_rate",
            "active_low_share",
            "routing_targeted_volume",
        )
    }

    timed_rows = records["response_timed"]
    counterfactual_keys = tuple(counterfactual_diagnostics[0]) if counterfactual_diagnostics else ()
    return {
        "seed": seed,
        "methods": methods,
        "training_dose_exact": bool(all(training_dose_exact)),
        "eval_dose_exact": bool(all(eval_dose_exact)),
        "counterfactual_response_diagnostic": {
            key: float(np.mean([row[key] for row in counterfactual_diagnostics]))
            for key in counterfactual_keys
        },
        "schedule_diagnostic": {
            "timed_activation_rate": float(np.mean([bool(row["gate_active"]) for row in timed_rows])),
        },
    }


def aggregate(seed_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    method_names = (
        "reference_no_intervention",
        "response_timed",
        "response_yoked",
        "response_gate_off",
    )
    methods: Dict[str, object] = {}
    for method in method_names:
        blocks = [dict(dict(result["methods"])[method]) for result in seed_results]
        keys = sorted({key for block in blocks for key in dict(block["eval_all"])})
        methods[method] = {
            "eval_all": {
                key: mean_se([float(dict(block["eval_all"]).get(key, 0.0)) for block in blocks])
                for key in keys
            }
        }

    def risk_metric(result: Dict[str, object], method: str, risk: str, key: str) -> float:
        block = dict(dict(result["methods"])[method])
        return float(dict(dict(block["eval_by_risk"])[risk])[key])

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
        risk_metric(result, "reference_no_intervention", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    timed_rewards = [
        float(dict(dict(dict(result["methods"])["response_timed"])["eval_all"])["mean_reward_mean"])
        for result in seed_results
    ]
    yoked_rewards = [
        float(dict(dict(dict(result["methods"])["response_yoked"])["eval_all"])["mean_reward_mean"])
        for result in seed_results
    ]
    response_gaps = [
        float(dict(result["counterfactual_response_diagnostic"])[
            "low_risk_high_spread_half_spread_reduction"
        ])
        for result in seed_results
    ]
    response_change_rates = [
        float(dict(result["counterfactual_response_diagnostic"])["action_change_rate"])
        for result in seed_results
    ]
    response_cap_gains = [
        float(dict(result["counterfactual_response_diagnostic"])[
            "low_risk_cap_compliance_gain"
        ])
        for result in seed_results
    ]
    activation = float(np.mean([
        float(dict(result["schedule_diagnostic"])["timed_activation_rate"])
        for result in seed_results
    ]))
    checks = {
        "training_dose_exact": bool(all(bool(result["training_dose_exact"]) for result in seed_results)),
        "eval_dose_exact": bool(all(bool(result["eval_dose_exact"]) for result in seed_results)),
        "no_intervention_failure_rate_at_least_70pct": bool(np.mean(np.asarray(no_low) >= 4.0) >= 0.70),
        "eval_activation_in_5_to_40pct": bool(0.05 <= activation <= 0.40),
        "counterfactual_signal_reduces_low_risk_high_spread_by_0_3_tick": bool(
            float(np.mean(response_gaps)) >= 0.3
        ),
        "counterfactual_response_positive_in_at_least_70pct_seeds": bool(
            np.mean(np.asarray(response_gaps) > 0.0) >= 0.70
        ),
        "timed_beats_yoked_on_low_risk_spread": bool(float(np.mean(low_deltas)) > 0.0),
        "paired_positive_seed_rate_at_least_70pct": bool(np.mean(np.asarray(low_deltas) > 0.0) >= 0.70),
        "high_risk_spread_not_reduced_more_than_half_tick": bool(float(np.mean(high_deltas)) <= 0.5),
        "reward_not_collapsed_vs_yoked": bool(float(np.mean(timed_rewards)) >= float(np.mean(yoked_rewards)) - 0.005),
    }
    return {
        "methods": methods,
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
        "counterfactual_signal_response": {
            **mean_se(response_gaps),
            "positive_seed_count": int(np.sum(np.asarray(response_gaps) > 0.0)),
            "n_seeds": len(response_gaps),
            "values": response_gaps,
            "action_change_rate": mean_se(response_change_rates),
            "low_risk_cap_compliance_gain": mean_se(response_cap_gains),
        },
        "pass_checks": checks,
        "passes_all": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--response-train-rate", type=float, default=0.50)
    parser.add_argument(
        "--response-train-low-risk-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--risk-stratified-yoke",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--schedule-from-response-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--threshold-spread", type=float, default=4.0)
    parser.add_argument("--activation-cap", type=float, default=0.35)
    parser.add_argument("--routing-rho", type=float, default=0.75)
    parser.add_argument("--routing-routable-fraction", type=float, default=0.70)
    parser.add_argument(
        "--routing-platform-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--routing-target-kappa", type=float, default=4.0)
    parser.add_argument(
        "--routing-eligibility-mode",
        choices=["relative", "absolute_cap", "soft_rank"],
        default="absolute_cap",
    )
    parser.add_argument("--gate-config", default=None)
    parser.add_argument("--warm-start-steps", type=int, default=200)
    parser.add_argument("--warm-start-updates", type=int, default=10)
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
        default="revision/experiments/outputs/lob_mappo_frozen_policy_timing_pilot_v1",
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
        routing_platform_balanced=args.routing_platform_balanced,
        routing_relative_improvement_ticks=0.5,
        routing_target_kappa=args.routing_target_kappa,
        routing_eligibility_mode=args.routing_eligibility_mode,
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
        print(f"seed={seed} frozen-policy timing", flush=True)
        seed_results.append(run_seed(
            replace(config, seed=seed),
            mappo,
            gate_config,
            args.routing_rho,
            args.response_train_rate,
            args.episodes,
            args.eval_episodes,
            seed,
            args.warm_start_steps,
            args.warm_start_updates,
            args.warm_start_low_half_spread,
            args.warm_start_high_half_spread,
            args.response_train_low_risk_only,
            args.risk_stratified_yoke,
            args.schedule_from_response_policy,
        ))

    summary = {
        "metadata": {
            "stage": "local_lob_mappo_frozen_responsive_policy_timing_pilot_not_full_main",
            "device": device,
            "seeds": seeds,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "response_train_rate": args.response_train_rate,
            "response_train_low_risk_only": args.response_train_low_risk_only,
            "risk_stratified_yoke": args.risk_stratified_yoke,
            "schedule_from_response_policy": args.schedule_from_response_policy,
            "schedule_uses_gate_off_potential_outcome": True,
            "routing_rho": args.routing_rho,
            "same_response_policy_for_timed_and_yoked_eval": True,
            "config": asdict(config),
            "gate_config": asdict(gate_config),
            "mappo_config": asdict(mappo),
        },
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "paired_low_risk": summary["aggregate"]["paired_yoked_minus_timed_low_risk_spread"],
        "counterfactual_response": summary["aggregate"]["counterfactual_signal_response"],
        "pass_checks": summary["aggregate"]["pass_checks"],
        "passes_all": summary["aggregate"]["passes_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
