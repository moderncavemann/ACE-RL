#!/usr/bin/env python3
"""CRN and dose-yoked timing pilot for the redesigned local LOB assay.

The primary endpoint is gate-off evaluation after training. The yoked arm
replays a within-episode permutation of the timed arm's activation schedule, so
intervention dose is identical and only state-contingent timing is destroyed.
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

from run_lob_mappo_pilot import (  # noqa: E402
    agent_observations,
    choose_device,
    critic_observations,
    evaluation_episode_seed,
    evaluate_policy,
    warm_start_risk_policy,
)
from run_lob_marl_pilot import (  # noqa: E402
    load_lob_gate_config,
    parse_seeds,
    summarize_by_risk,
    summarize_records,
)
from src.agents.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from src.envs.local_lob_market_env import LocalLOBConfig, LocalLOBMarketEnv  # noqa: E402
from src.interventions.lob_gates import LOBGateConfig, LOBGateMode, LOBInterventionGate  # noqa: E402


TIMING_MODES: Dict[str, LOBGateMode] = {
    "oracle_controlled": "oracle_low_high_spread_controlled",
    "risk_aware_controlled": "risk_aware_controlled",
}


def mean_se(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def episode_risk_path(config: LocalLOBConfig, seed: int) -> list[str]:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    actions = np.zeros(config.n_agents, dtype=np.int64)
    risk_path: list[str] = []
    done = False
    while not done:
        risk_path.append(str(env.risk_state))
        _, _, done, _, _ = env.step(actions)
    return risk_path


def risk_stratified_yoked_schedule(
    schedule: Sequence[bool],
    risk_path: Sequence[str],
    seed: int,
) -> list[bool]:
    if len(schedule) != len(risk_path):
        raise ValueError("schedule and risk path must have the same length")
    if any(bool(active) and risk != "low" for active, risk in zip(schedule, risk_path)):
        raise ValueError("timed source schedule must activate only in low risk")
    low_indices = np.asarray(
        [index for index, risk in enumerate(risk_path) if risk == "low"],
        dtype=int,
    )
    values = np.asarray([bool(schedule[index]) for index in low_indices], dtype=bool)
    np.random.default_rng(seed).shuffle(values)
    yoked = np.zeros(len(schedule), dtype=bool)
    yoked[low_indices] = values
    return yoked.tolist()


def make_trainer(config: LocalLOBConfig, base: MAPPOConfig, seed: int) -> MAPPOTrainer:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    obs = agent_observations(env, include_gate_signal=True)
    critic_obs = critic_observations(env, obs)
    return MAPPOTrainer(MAPPOConfig(**{
        **asdict(base),
        "obs_dim": int(obs.shape[1]),
        "critic_obs_dim": int(critic_obs.shape[1]),
        "n_actions": config.n_actions,
        "n_agents": config.n_agents,
        "seed": seed,
    }))


def make_warm_trainer(
    config: LocalLOBConfig,
    base: MAPPOConfig,
    seed: int,
    warm_steps: int,
    warm_updates: int,
    warm_low: int,
    warm_high: int,
) -> tuple[MAPPOTrainer, Dict[str, float]]:
    trainer = make_trainer(config, base, seed)
    env = LocalLOBMarketEnv(config)
    warm = warm_start_risk_policy(
        env,
        trainer,
        warm_low,
        warm_high,
        warm_steps,
        warm_updates,
        seed=seed + 500_000,
        include_gate_signal=True,
    )
    return trainer, warm


def train_episode(
    env: LocalLOBMarketEnv,
    trainer: MAPPOTrainer,
    gate: LOBInterventionGate,
    routing_rho: float,
    seed: int,
    active_schedule: Sequence[bool] | None = None,
) -> tuple[Dict[str, float], list[bool]]:
    env.reset(seed=seed)
    gate.reset()
    obs_rows: list[np.ndarray] = []
    critic_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    logp_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    schedule: list[bool] = []
    active_low: list[float] = []
    targeted_volume: list[float] = []
    done = False
    t = 0
    while not done:
        decision = gate.decide(
            env.risk_state,
            env.last_agent_spreads,
            env.last_flow_shares,
            env.last_total_volume,
        )
        active = bool(decision["active"]) if active_schedule is None else bool(active_schedule[t])
        obs = agent_observations(
            env,
            include_gate_signal=True,
            gate_signal=float(active),
        )
        critic_obs = critic_observations(env, obs)
        actions, logp, values = trainer.act(obs, critic_obs, deterministic=False)
        boost = routing_rho if active else 0.0
        _, rewards, done, _, info = env.step(
            {i: int(actions[i]) for i in range(env.config.n_agents)},
            routing_flow_boost=boost,
            gate_active=active,
            gate_score=float(decision["score"]),
        )
        obs_rows.append(obs)
        critic_rows.append(critic_obs)
        action_rows.append(actions.astype(np.int64))
        logp_rows.append(logp.astype(float))
        value_rows.append(values.astype(float))
        reward_rows.append(np.array([rewards[i] for i in range(env.config.n_agents)], dtype=float))
        schedule.append(active)
        active_low.append(float(active and str(info["risk_state"]) == "low"))
        targeted_volume.append(float(info.get("routing_targeted_volume", 0.0)))
        t += 1

    stats = trainer.update_episode(
        np.stack(obs_rows),
        np.stack(critic_rows),
        np.stack(action_rows),
        np.stack(logp_rows),
        np.stack(reward_rows),
        np.stack(value_rows),
    )
    activation = float(np.mean(schedule))
    stats.update({
        "gate_activation_rate": activation,
        "active_low_share": float(np.sum(active_low) / max(np.sum(schedule), 1.0)),
        "routing_targeted_volume": float(np.mean(targeted_volume)),
    })
    return stats, schedule


def reference_schedule(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    gate_config: LOBGateConfig,
    timing_mode: LOBGateMode,
    seed: int,
) -> list[bool]:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    gate = LOBInterventionGate(config, gate_config, timing_mode)
    gate.reset()
    schedule: list[bool] = []
    done = False
    while not done:
        decision = gate.decide(
            env.risk_state,
            env.last_agent_spreads,
            env.last_flow_shares,
            env.last_total_volume,
        )
        obs = agent_observations(env, include_gate_signal=True, gate_signal=0.0)
        critic_obs = critic_observations(env, obs)
        actions, _, _ = trainer.act(obs, critic_obs, deterministic=True)
        _, _, done, _, _ = env.step(
            {i: int(actions[i]) for i in range(config.n_agents)},
            routing_flow_boost=0.0,
            gate_active=False,
            gate_score=float(decision["score"]),
        )
        schedule.append(bool(decision["active"]))
    return schedule


def evaluate_scheduled_policy(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    schedule: Sequence[bool],
    routing_rho: float,
    seed: int,
    method: str,
) -> List[Dict[str, object]]:
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    records: List[Dict[str, object]] = []
    done = False
    t = 0
    while not done:
        active = bool(schedule[t])
        obs = agent_observations(
            env,
            include_gate_signal=True,
            gate_signal=float(active),
        )
        critic_obs = critic_observations(env, obs)
        actions, _, _ = trainer.act(obs, critic_obs, deterministic=True)
        _, _, done, _, info = env.step(
            {i: int(actions[i]) for i in range(config.n_agents)},
            routing_flow_boost=routing_rho if active else 0.0,
            gate_active=active,
            gate_score=float(active),
        )
        row = {
            key: value for key, value in info.items()
            if isinstance(value, (str, int, float, bool, np.integer, np.floating))
        }
        row.update({"method": method, "seed": seed, "gate_active": active})
        records.append(row)
        t += 1
    return records


def summarize_method(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    seed: int,
    eval_episodes: int,
    method: str,
) -> Dict[str, object]:
    env = LocalLOBMarketEnv(config)
    gate = LOBInterventionGate(config, LOBGateConfig(), "none")
    records = evaluate_policy(
        env,
        trainer,
        gate,
        routing_rho=0.0,
        method=method,
        seed=seed,
        eval_episodes=eval_episodes,
        gate_off=True,
        include_gate_signal=True,
    )
    return {
        "eval_all": summarize_records(records),
        "eval_by_risk": summarize_by_risk(records),
    }


def run_seed(
    config: LocalLOBConfig,
    mappo: MAPPOConfig,
    gate_config: LOBGateConfig,
    timing_mode: LOBGateMode,
    routing_rho: float,
    episodes: int,
    eval_episodes: int,
    seed: int,
    warm_steps: int,
    warm_updates: int,
    warm_low: int,
    warm_high: int,
    training_schedule_source: str,
) -> Dict[str, object]:
    no_trainer, no_warm = make_warm_trainer(
        config, mappo, seed, warm_steps, warm_updates, warm_low, warm_high
    )
    timed_trainer, timed_warm = make_warm_trainer(
        config, mappo, seed, warm_steps, warm_updates, warm_low, warm_high
    )
    yoked_trainer, yoked_warm = make_warm_trainer(
        config, mappo, seed, warm_steps, warm_updates, warm_low, warm_high
    )
    no_env = LocalLOBMarketEnv(config)
    timed_env = LocalLOBMarketEnv(config)
    yoked_env = LocalLOBMarketEnv(config)
    no_gate = LOBInterventionGate(config, gate_config, "none")
    timed_gate = LOBInterventionGate(
        config,
        gate_config,
        timing_mode if training_schedule_source == "timed_online" else "none",
    )
    yoked_gate = LOBInterventionGate(config, gate_config, "none")
    training: Dict[str, List[Dict[str, float]]] = {
        "no_intervention": [],
        "timed": [],
        "yoked_random": [],
    }
    dose_exact: list[bool] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        reference_timed_schedule = None
        if training_schedule_source == "reference_policy":
            reference_timed_schedule = reference_schedule(
                config,
                no_trainer,
                gate_config,
                timing_mode,
                episode_seed,
            )
        no_stats, _ = train_episode(
            no_env, no_trainer, no_gate, 0.0, episode_seed
        )
        if training_schedule_source == "reference_policy":
            if reference_timed_schedule is None:
                raise RuntimeError("missing reference-policy timing schedule")
            timed_schedule = reference_timed_schedule
            timed_stats, replayed_timed_schedule = train_episode(
                timed_env,
                timed_trainer,
                timed_gate,
                routing_rho,
                episode_seed,
                active_schedule=timed_schedule,
            )
        else:
            timed_stats, timed_schedule = train_episode(
                timed_env,
                timed_trainer,
                timed_gate,
                routing_rho,
                episode_seed,
            )
            replayed_timed_schedule = timed_schedule
        yoked_schedule = risk_stratified_yoked_schedule(
            timed_schedule,
            episode_risk_path(config, episode_seed),
            seed=seed * 1_000_003 + episode,
        )
        yoked_stats, replayed_schedule = train_episode(
            yoked_env,
            yoked_trainer,
            yoked_gate,
            routing_rho,
            episode_seed,
            active_schedule=yoked_schedule,
        )
        dose_exact.append(
            sum(timed_schedule)
            == sum(replayed_timed_schedule)
            == sum(replayed_schedule)
        )
        training["no_intervention"].append(no_stats)
        training["timed"].append(timed_stats)
        training["yoked_random"].append(yoked_stats)

    trainers = {
        "no_intervention": no_trainer,
        "timed": timed_trainer,
        "yoked_random": yoked_trainer,
    }
    warm = {
        "no_intervention": no_warm,
        "timed": timed_warm,
        "yoked_random": yoked_warm,
    }
    methods: Dict[str, object] = {}
    for method, trainer in trainers.items():
        block = summarize_method(config, trainer, seed, eval_episodes, method)
        rows = training[method]
        block["training"] = {
            key: float(np.mean([row[key] for row in rows])) if rows else 0.0
            for key in (
                "entropy",
                "return_mean",
                "gate_activation_rate",
                "active_low_share",
                "routing_targeted_volume",
            )
        }
        block["warm_start"] = warm[method]
        methods[method] = block

    online_records: Dict[str, List[Dict[str, object]]] = {
        "no_intervention": [],
        "timed": [],
        "yoked_random": [],
    }
    eval_dose_exact: list[bool] = []
    for episode in range(eval_episodes):
        eval_seed = evaluation_episode_seed(seed, episode)
        timed_schedule = reference_schedule(
            config,
            no_trainer,
            gate_config,
            timing_mode,
            eval_seed,
        )
        yoked_schedule = risk_stratified_yoked_schedule(
            timed_schedule,
            episode_risk_path(config, eval_seed),
            seed=seed * 2_000_003 + 100_000 + episode,
        )
        eval_dose_exact.append(sum(timed_schedule) == sum(yoked_schedule))
        online_records["no_intervention"].extend(evaluate_scheduled_policy(
            config,
            no_trainer,
            [False] * len(timed_schedule),
            0.0,
            eval_seed,
            "no_intervention",
        ))
        online_records["timed"].extend(evaluate_scheduled_policy(
            config,
            timed_trainer,
            timed_schedule,
            routing_rho,
            eval_seed,
            "timed",
        ))
        online_records["yoked_random"].extend(evaluate_scheduled_policy(
            config,
            yoked_trainer,
            yoked_schedule,
            routing_rho,
            eval_seed,
            "yoked_random",
        ))
    for method, records in online_records.items():
        block = dict(methods[method])
        block["eval_online"] = summarize_records(records)
        block["eval_online_by_risk"] = summarize_by_risk(records)
        methods[method] = block
    return {
        "seed": seed,
        "methods": methods,
        "dose_exact_all_episodes": bool(all(dose_exact)),
        "eval_dose_exact_all_episodes": bool(all(eval_dose_exact)),
    }


def aggregate(seed_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    methods_out: Dict[str, object] = {}
    for method in ("no_intervention", "timed", "yoked_random"):
        rows = [dict(dict(result["methods"])[method]) for result in seed_results]
        eval_all_keys = sorted({key for row in rows for key in dict(row["eval_all"])})
        eval_online_keys = sorted({key for row in rows for key in dict(row["eval_online"])})
        train_keys = sorted({key for row in rows for key in dict(row["training"])})
        methods_out[method] = {
            "eval_all": {
                key: mean_se([float(dict(row["eval_all"]).get(key, 0.0)) for row in rows])
                for key in eval_all_keys
            },
            "eval_online": {
                key: mean_se([float(dict(row["eval_online"]).get(key, 0.0)) for row in rows])
                for key in eval_online_keys
            },
            "training": {
                key: mean_se([float(dict(row["training"]).get(key, 0.0)) for row in rows])
                for key in train_keys
            },
        }

    def metric(
        result: Dict[str, object],
        method: str,
        risk: str,
        key: str,
        block_name: str = "eval_online_by_risk",
    ) -> float:
        block = dict(dict(result["methods"])[method])
        return float(dict(dict(block[block_name])[risk]).get(key, 0.0))

    online_low_deltas = [
        metric(result, "yoked_random", "low", "avg_agent_spread_mean")
        - metric(result, "timed", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    online_high_deltas = [
        metric(result, "yoked_random", "high", "avg_agent_spread_mean")
        - metric(result, "timed", "high", "avg_agent_spread_mean")
        for result in seed_results
    ]
    no_low = [
        metric(
            result,
            "no_intervention",
            "low",
            "avg_agent_spread_mean",
            block_name="eval_by_risk",
        )
        for result in seed_results
    ]
    timed_rewards = [
        float(dict(dict(dict(result["methods"])["timed"])["eval_all"])["mean_reward_mean"])
        for result in seed_results
    ]
    yoked_rewards = [
        float(dict(dict(dict(result["methods"])["yoked_random"])["eval_all"])["mean_reward_mean"])
        for result in seed_results
    ]
    activation = float(dict(dict(methods_out["timed"])["training"])["gate_activation_rate"]["mean"])
    gate_off_low_deltas = [
        metric(
            result,
            "yoked_random",
            "low",
            "avg_agent_spread_mean",
            block_name="eval_by_risk",
        )
        - metric(
            result,
            "timed",
            "low",
            "avg_agent_spread_mean",
            block_name="eval_by_risk",
        )
        for result in seed_results
    ]
    gate_off_high_deltas = [
        metric(
            result,
            "yoked_random",
            "high",
            "avg_agent_spread_mean",
            block_name="eval_by_risk",
        )
        - metric(
            result,
            "timed",
            "high",
            "avg_agent_spread_mean",
            block_name="eval_by_risk",
        )
        for result in seed_results
    ]
    pass_checks = {
        "dose_exact_every_episode": bool(all(bool(result["dose_exact_all_episodes"]) for result in seed_results)),
        "eval_dose_exact_every_episode": bool(
            all(bool(result["eval_dose_exact_all_episodes"]) for result in seed_results)
        ),
        "no_intervention_failure_rate_at_least_70pct": bool(np.mean(np.asarray(no_low) >= 4.0) >= 0.70),
        "timed_activation_in_5_to_40pct": bool(0.05 <= activation <= 0.40),
        "timed_training_beats_yoked_on_gate_off_low_risk_spread": bool(
            float(np.mean(gate_off_low_deltas)) > 0.0
        ),
        "gate_off_paired_positive_seed_rate_at_least_70pct": bool(
            np.mean(np.asarray(gate_off_low_deltas) > 0.0) >= 0.70
        ),
        "gate_off_high_risk_spread_not_reduced_more_than_half_tick": bool(
            float(np.mean(gate_off_high_deltas)) <= 0.5
        ),
        "reward_not_collapsed_vs_yoked": bool(float(np.mean(timed_rewards)) >= float(np.mean(yoked_rewards)) - 0.005),
    }
    return {
        "methods": methods_out,
        "primary_gate_off_yoked_minus_timed_low_risk_spread": {
            **mean_se(gate_off_low_deltas),
            "positive_seed_count": int(np.sum(np.asarray(gate_off_low_deltas) > 0.0)),
            "n_seeds": len(gate_off_low_deltas),
            "values": gate_off_low_deltas,
        },
        "primary_gate_off_yoked_minus_timed_high_risk_spread": {
            **mean_se(gate_off_high_deltas),
            "values": gate_off_high_deltas,
        },
        "secondary_online_yoked_minus_timed_low_risk_spread": {
            **mean_se(online_low_deltas),
            "positive_seed_count": int(np.sum(np.asarray(online_low_deltas) > 0.0)),
            "n_seeds": len(online_low_deltas),
            "values": online_low_deltas,
        },
        "secondary_online_yoked_minus_timed_high_risk_spread": {
            **mean_se(online_high_deltas),
            "values": online_high_deltas,
        },
        "no_intervention_low_risk_spreads": no_low,
        "pass_checks": pass_checks,
        "passes_all": bool(all(pass_checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--timing-mode", choices=sorted(TIMING_MODES), default="oracle_controlled")
    parser.add_argument(
        "--training-schedule-source",
        choices=["reference_policy", "timed_online"],
        default="timed_online",
    )
    parser.add_argument("--gate-config", default=None)
    parser.add_argument("--threshold-spread", type=float, default=4.0)
    parser.add_argument("--activation-cap", type=float, default=0.35)
    parser.add_argument("--routing-rho", type=float, default=1.0)
    parser.add_argument("--routing-routable-fraction", type=float, default=0.90)
    parser.add_argument(
        "--routing-platform-balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--routing-relative-improvement-ticks", type=float, default=0.5)
    parser.add_argument("--routing-target-kappa", type=float, default=4.0)
    parser.add_argument("--p-low-to-high", type=float, default=0.08)
    parser.add_argument("--p-high-to-low", type=float, default=0.08)
    parser.add_argument("--warm-start-steps", type=int, default=200)
    parser.add_argument("--warm-start-updates", type=int, default=20)
    parser.add_argument("--warm-start-low-half-spread", type=int, default=3)
    parser.add_argument("--warm-start-high-half-spread", type=int, default=6)
    parser.add_argument("--policy-mode", choices=["shared", "multihead", "independent"], default="multihead")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="revision/experiments/outputs/lob_mappo_yoked_oracle_pilot_v1",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    config = LocalLOBConfig(
        n_agents=args.n_agents,
        episode_length=args.episode_length,
        inventory_penalty=0.0001,
        base_flow_intensity=2.0,
        routing_mode="platform_reallocate",
        routing_routable_fraction=args.routing_routable_fraction,
        routing_platform_balanced=args.routing_platform_balanced,
        routing_relative_improvement_ticks=args.routing_relative_improvement_ticks,
        routing_target_kappa=args.routing_target_kappa,
        routing_eligibility_mode="absolute_cap",
        routing_target_half_spread_cap=3.0,
        risk_coupled_economics=True,
        mark_to_fundamental=True,
        informed_flow_prob=0.7,
        spread_elasticity=0.14,
        low_sigma=0.005,
        high_sigma=0.08,
        p_low_to_high=args.p_low_to_high,
        p_high_to_low=args.p_high_to_low,
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
        policy_mode=args.policy_mode,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    seed_results: List[Dict[str, object]] = []
    timing_mode = TIMING_MODES[args.timing_mode]
    for seed in seeds:
        print(f"seed={seed} timing={args.timing_mode}", flush=True)
        seed_results.append(run_seed(
            replace(config, seed=seed),
            mappo,
            gate_config,
            timing_mode,
            args.routing_rho,
            args.episodes,
            args.eval_episodes,
            seed,
            args.warm_start_steps,
            args.warm_start_updates,
            args.warm_start_low_half_spread,
            args.warm_start_high_half_spread,
            args.training_schedule_source,
        ))

    summary = {
        "metadata": {
            "stage": "local_lob_mappo_yoked_timing_pilot_not_full_main",
            "device": device,
            "seeds": seeds,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "timing_mode": args.timing_mode,
            "training_schedule_source": args.training_schedule_source,
            "routing_rho": args.routing_rho,
            "reference_policy_schedules_training_before_each_update": bool(
                args.training_schedule_source == "reference_policy"
            ),
            "timed_online_schedule_is_dose_yoked_to_random": bool(
                args.training_schedule_source == "timed_online"
            ),
            "risk_stratified_dose_yoking": True,
            "gate_off_persistence_primary_eval": True,
            "frozen_schedule_online_secondary_eval": True,
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
        "paired_low_risk": summary["aggregate"][
            "primary_gate_off_yoked_minus_timed_low_risk_spread"
        ],
        "pass_checks": summary["aggregate"]["pass_checks"],
        "passes_all": summary["aggregate"]["passes_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
