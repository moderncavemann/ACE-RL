#!/usr/bin/env python3
"""Screen a frozen learned LOB gate against exact-dose matched random timing.

The gate schedule is generated from a passive CRN reference path. The matched
random arm permutes that schedule within each episode and risk regime, keeping
the intervention dose exactly fixed while destroying state-contingent timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lob_mappo_contract_response_pilot import (  # noqa: E402
    calibrate_low_risk_threshold,
    make_response_adapter,
    mean_training,
)
from run_lob_mappo_frozen_policy_timing_pilot import (  # noqa: E402
    counterfactual_signal_response,
    fixed_rate_low_risk_schedule,
)
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
from src.interventions.learned_lob_gate import (  # noqa: E402
    LearnedLOBGateModel,
    learned_gate_features,
    learned_gate_target,
)
from src.interventions.lob_gates import LOBGateConfig, LOBInterventionGate  # noqa: E402


def object_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


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


def risk_stratified_matched_schedule(
    schedule: Sequence[bool],
    risk_path: Sequence[str],
    seed: int,
) -> list[bool]:
    """Permute activations within risk strata, preserving exact stratum dose."""
    if len(schedule) != len(risk_path):
        raise ValueError("schedule and risk path must have the same length")
    rng = np.random.default_rng(seed)
    matched = np.zeros(len(schedule), dtype=bool)
    for risk in sorted(set(str(value) for value in risk_path)):
        indices = np.asarray([
            index for index, value in enumerate(risk_path) if str(value) == risk
        ], dtype=int)
        values = np.asarray([bool(schedule[index]) for index in indices], dtype=bool)
        rng.shuffle(values)
        matched[indices] = values
    return matched.tolist()


def risk_stratified_circular_schedule(
    schedule: Sequence[bool],
    risk_path: Sequence[str],
    seed: int,
) -> list[bool]:
    """Circularly shift each risk stratum while preserving its activation runs."""
    if len(schedule) != len(risk_path):
        raise ValueError("schedule and risk path must have the same length")
    rng = np.random.default_rng(seed)
    shifted = np.zeros(len(schedule), dtype=bool)
    for risk in sorted(set(str(value) for value in risk_path)):
        indices = np.asarray([
            index for index, value in enumerate(risk_path) if str(value) == risk
        ], dtype=int)
        values = np.asarray([bool(schedule[index]) for index in indices], dtype=bool)
        if len(values) > 1:
            offset = int(rng.integers(1, len(values)))
            values = np.roll(values, offset)
        shifted[indices] = values
    return shifted.tolist()


def risk_dose_exact(
    source: Sequence[bool],
    matched: Sequence[bool],
    risk_path: Sequence[str],
) -> bool:
    if not (len(source) == len(matched) == len(risk_path)):
        return False
    if sum(bool(value) for value in source) != sum(bool(value) for value in matched):
        return False
    return all(
        sum(bool(active) for active, state in zip(source, risk_path) if state == risk)
        == sum(bool(active) for active, state in zip(matched, risk_path) if state == risk)
        for risk in set(risk_path)
    )


def learned_reference_schedule(
    config: LocalLOBConfig,
    trainer: MAPPOTrainer,
    model: LearnedLOBGateModel,
    seed: int,
    action_namespace: int = 31_337,
) -> tuple[list[bool], List[Dict[str, object]]]:
    """Apply the frozen gate to a passive stochastic reference trajectory."""
    env = LocalLOBMarketEnv(config)
    env.reset(seed=seed)
    schedule: list[bool] = []
    diagnostics: List[Dict[str, object]] = []
    done = False
    while not done:
        features = learned_gate_features(env, model.expected_volume)
        decision = model.decide(features.tolist())
        active = bool(decision["active"])
        diagnostics.append({
            "risk_state": str(env.risk_state),
            "target": bool(learned_gate_target(env)),
            "active": active,
            "probability": float(decision["probability"]),
            "contract_severity": float(env.current_routing_contract_severity()),
        })
        obs = agent_observations(env, include_gate_signal=True, gate_signal=0.0)
        critic_obs = critic_observations(env, obs)
        action_uniforms = np.random.default_rng(np.random.SeedSequence([
            int(seed),
            int(env.t),
            int(action_namespace),
        ])).random(config.n_agents)
        actions, _, _ = trainer.act(
            obs,
            critic_obs,
            deterministic=False,
            action_uniforms=action_uniforms,
        )
        _, _, done, _, _ = env.step(
            {i: int(actions[i]) for i in range(config.n_agents)},
            routing_flow_boost=0.0,
            gate_active=False,
            gate_score=float(decision["probability"]),
        )
        schedule.append(active)
    return schedule, diagnostics


def schedule_counts(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    active = np.asarray([bool(row["active"]) for row in rows], dtype=bool)
    target = np.asarray([bool(row["target"]) for row in rows], dtype=bool)
    high = np.asarray([str(row["risk_state"]) == "high" for row in rows], dtype=bool)
    return {
        "n_steps": int(len(rows)),
        "active": int(np.sum(active)),
        "target_positive": int(np.sum(target)),
        "true_positive": int(np.sum(active & target)),
        "false_positive": int(np.sum(active & ~target)),
        "high_risk_steps": int(np.sum(high)),
        "high_risk_active": int(np.sum(active & high)),
    }


def rates_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    active = counts["active"]
    positive = counts["target_positive"]
    negative = counts["n_steps"] - positive
    return {
        "activation_rate": active / max(counts["n_steps"], 1),
        "target_prevalence": positive / max(counts["n_steps"], 1),
        "precision": counts["true_positive"] / max(active, 1),
        "recall": counts["true_positive"] / max(positive, 1),
        "negative_fpr": counts["false_positive"] / max(negative, 1),
        "high_risk_fpr": counts["high_risk_active"] / max(counts["high_risk_steps"], 1),
    }


def train_response_variant(
    config: LocalLOBConfig,
    response: MAPPOTrainer,
    gate_config: LOBGateConfig,
    routing_rho: float,
    response_episodes: int,
    response_train_rate: float,
    seed: int,
) -> tuple[List[Dict[str, float]], bool]:
    env = LocalLOBMarketEnv(config)
    no_gate = LOBInterventionGate(config, gate_config, "none")
    training: List[Dict[str, float]] = []
    schedules_exact: list[bool] = []
    for episode in range(response_episodes):
        episode_seed = seed + 100_000 + episode
        schedule = fixed_rate_low_risk_schedule(
            config,
            episode_seed,
            response_train_rate,
            schedule_seed=seed * 3_000_017 + episode,
        )
        stats, replayed = train_episode(
            env,
            response,
            no_gate,
            routing_rho,
            episode_seed,
            active_schedule=schedule,
            announce_contract_strength=True,
        )
        training.append(stats)
        schedules_exact.append(schedule == replayed)
    return training, bool(all(schedules_exact))


def run_seed(
    config: LocalLOBConfig,
    mappo: MAPPOConfig,
    gate_config: LOBGateConfig,
    model: LearnedLOBGateModel,
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
    include_full_main: bool = False,
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
        routing_excess_reference_half_spread=float(calibration["threshold_spread"]),
        routing_excess_scale_ticks=float(calibration["contract_scale_ticks"]),
    )

    response = make_response_adapter(contract_config, mappo, base, seed)
    response_training, response_schedule_exact = train_response_variant(
        contract_config,
        response,
        gate_config,
        routing_rho,
        response_episodes,
        response_train_rate,
        seed,
    )

    fixed_config = replace(contract_config, routing_contract="fixed_share")
    fixed_response: MAPPOTrainer | None = None
    no_routing_response: MAPPOTrainer | None = None
    fixed_training: List[Dict[str, float]] = []
    no_routing_training: List[Dict[str, float]] = []
    fixed_schedule_exact = True
    no_routing_schedule_exact = True
    if include_full_main:
        fixed_response = make_response_adapter(fixed_config, mappo, base, seed)
        fixed_training, fixed_schedule_exact = train_response_variant(
            fixed_config,
            fixed_response,
            gate_config,
            routing_rho,
            response_episodes,
            response_train_rate,
            seed,
        )
        no_routing_response = make_response_adapter(
            contract_config,
            mappo,
            base,
            seed,
        )
        no_routing_training, no_routing_schedule_exact = train_response_variant(
            contract_config,
            no_routing_response,
            gate_config,
            0.0,
            response_episodes,
            response_train_rate,
            seed,
        )

    records: Dict[str, List[Dict[str, object]]] = {
        "base_no_intervention": [],
        "learned_gate": [],
        "matched_random": [],
        "oracle_gate": [],
        "response_gate_off": [],
    }
    if include_full_main:
        records.update({
            "circular_shuffled": [],
            "threshold_fixed_pool": [],
            "always_on_fixed_pool": [],
            "no_routing_trained": [],
            "no_routing_gate_off": [],
            "no_state_pool_learned": [],
            "no_state_pool_matched": [],
            "no_state_pool_gate_off": [],
        })
    dose_exact: list[bool] = []
    circular_dose_exact: list[bool] = []
    gate_off_equivalent: list[bool] = []
    ablation_gate_off_equivalent: list[bool] = []
    counterfactual_rows: list[Dict[str, float]] = []
    gate_rows: List[Dict[str, object]] = []
    oracle_activation: list[float] = []
    for episode in range(eval_episodes):
        eval_seed = evaluation_episode_seed(seed, episode)
        learned_schedule, episode_gate_rows = learned_reference_schedule(
            contract_config,
            base,
            model,
            eval_seed,
        )
        risk_path = [str(row["risk_state"]) for row in episode_gate_rows]
        matched_schedule = risk_stratified_matched_schedule(
            learned_schedule,
            risk_path,
            seed=seed * 4_000_037 + episode,
        )
        circular_schedule = risk_stratified_circular_schedule(
            learned_schedule,
            risk_path,
            seed=seed * 5_000_011 + episode,
        )
        threshold_schedule = [
            float(cast(Any, row["contract_severity"])) > 0.0
            for row in episode_gate_rows
        ]
        always_schedule = [True] * len(learned_schedule)
        oracle_schedule = reference_schedule(
            contract_config,
            base,
            calibrated_gate_config,
            "oracle_low_high_spread_controlled",
            eval_seed,
            stochastic_actions=True,
            action_namespace=31_337,
        )
        dose_exact.append(risk_dose_exact(
            learned_schedule,
            matched_schedule,
            risk_path,
        ))
        circular_dose_exact.append(risk_dose_exact(
            learned_schedule,
            circular_schedule,
            risk_path,
        ))
        oracle_activation.append(float(np.mean(oracle_schedule)))
        gate_rows.extend(episode_gate_rows)

        base_rows = evaluate_scheduled_policy(
            contract_config,
            base,
            [False] * len(learned_schedule),
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
            [False] * len(learned_schedule),
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
        for method, schedule in (
            ("learned_gate", learned_schedule),
            ("matched_random", matched_schedule),
            ("oracle_gate", oracle_schedule),
        ):
            records[method].extend(evaluate_scheduled_policy(
                contract_config,
                response,
                schedule,
                routing_rho,
                eval_seed,
                method,
                stochastic_actions=True,
                action_namespace=31_337,
                announce_contract_strength=True,
            ))
        if include_full_main:
            if fixed_response is None or no_routing_response is None:
                raise RuntimeError("full-main ablation policies were not trained")
            records["circular_shuffled"].extend(evaluate_scheduled_policy(
                contract_config,
                response,
                circular_schedule,
                routing_rho,
                eval_seed,
                "circular_shuffled",
                stochastic_actions=True,
                action_namespace=31_337,
                announce_contract_strength=True,
            ))
            no_routing_gate_off_rows = evaluate_scheduled_policy(
                contract_config,
                no_routing_response,
                [False] * len(learned_schedule),
                0.0,
                eval_seed,
                "no_routing_gate_off",
                stochastic_actions=True,
                action_namespace=31_337,
                announce_contract_strength=True,
            )
            no_state_gate_off_rows = evaluate_scheduled_policy(
                fixed_config,
                fixed_response,
                [False] * len(learned_schedule),
                0.0,
                eval_seed,
                "no_state_pool_gate_off",
                stochastic_actions=True,
                action_namespace=31_337,
                announce_contract_strength=True,
            )
            ablation_gate_off_equivalent.append(all(
                np.isclose(
                    float(base_row["avg_agent_spread"]),
                    float(ablation_row["avg_agent_spread"]),
                    atol=0.0,
                    rtol=0.0,
                )
                for base_row, ablation_row in zip(
                    base_rows,
                    no_routing_gate_off_rows,
                )
            ) and all(
                np.isclose(
                    float(base_row["avg_agent_spread"]),
                    float(ablation_row["avg_agent_spread"]),
                    atol=0.0,
                    rtol=0.0,
                )
                for base_row, ablation_row in zip(
                    base_rows,
                    no_state_gate_off_rows,
                )
            ))
            records["no_routing_gate_off"].extend(no_routing_gate_off_rows)
            records["no_state_pool_gate_off"].extend(no_state_gate_off_rows)
            records["no_routing_trained"].extend(evaluate_scheduled_policy(
                contract_config,
                no_routing_response,
                learned_schedule,
                0.0,
                eval_seed,
                "no_routing_trained",
                stochastic_actions=True,
                action_namespace=31_337,
                announce_contract_strength=True,
            ))
            for method, schedule in (
                ("no_state_pool_learned", learned_schedule),
                ("no_state_pool_matched", matched_schedule),
                ("threshold_fixed_pool", threshold_schedule),
                ("always_on_fixed_pool", always_schedule),
            ):
                records[method].extend(evaluate_scheduled_policy(
                    fixed_config,
                    fixed_response,
                    schedule,
                    routing_rho,
                    eval_seed,
                    method,
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
    if include_full_main:
        methods["no_routing_gate_off"]["training"] = mean_training(
            no_routing_training
        )
        methods["no_state_pool_gate_off"]["training"] = mean_training(
            fixed_training
        )
    diagnostic_keys = tuple(counterfactual_rows[0]) if counterfactual_rows else ()
    counts = schedule_counts(gate_rows)
    return {
        "seed": seed,
        "methods": methods,
        "training_schedule_exact": bool(
            response_schedule_exact
            and fixed_schedule_exact
            and no_routing_schedule_exact
        ),
        "eval_risk_stratified_dose_exact": bool(all(dose_exact)),
        "eval_circular_risk_stratified_dose_exact": bool(all(circular_dose_exact)),
        "gate_off_base_equivalent": bool(all(gate_off_equivalent)),
        "ablation_gate_off_base_equivalent": bool(
            all(ablation_gate_off_equivalent)
        ),
        "calibration": calibration,
        "learned_gate_counts": counts,
        "learned_gate_rates": rates_from_counts(counts),
        "oracle_activation_rate": float(np.mean(oracle_activation)),
        "counterfactual_response_diagnostic": {
            key: float(np.mean([row[key] for row in counterfactual_rows]))
            for key in diagnostic_keys
        },
    }


def aggregate(
    seed_results: Sequence[Dict[str, object]],
    provenance_disjoint: bool,
    min_expected_response: float,
    min_positive_seed_rate: float,
    min_activation: float,
    max_activation: float,
    max_high_risk_fpr: float,
    min_target_recall: float,
) -> Dict[str, object]:
    def method_block(
        result: Dict[str, object],
        method: str,
    ) -> Mapping[str, Any]:
        methods_block = object_mapping(result["methods"], "methods")
        return object_mapping(methods_block[method], f"method {method}")

    def all_block(
        result: Dict[str, object],
        method: str,
    ) -> Mapping[str, Any]:
        return object_mapping(method_block(result, method)["eval_all"], "eval_all")

    def risk_block(
        result: Dict[str, object],
        method: str,
        risk: str,
    ) -> Mapping[str, Any]:
        by_risk = object_mapping(
            method_block(result, method)["eval_by_risk"],
            "eval_by_risk",
        )
        return object_mapping(by_risk[risk], f"risk {risk}")

    def risk_metric(result: Dict[str, object], method: str, risk: str, key: str) -> float:
        return float(risk_block(result, method, risk)[key])

    def all_metric(result: Dict[str, object], method: str, key: str) -> float:
        return float(all_block(result, method)[key])

    if not seed_results:
        raise ValueError("at least one seed result is required")
    method_names = tuple(
        str(method)
        for method in object_mapping(
            seed_results[0]["methods"],
            "methods",
        )
    )
    methods: Dict[str, object] = {}
    for method in method_names:
        all_keys = sorted({
            key
            for result in seed_results
            for key in all_block(result, method)
        })
        risk_blocks: Dict[str, object] = {}
        for risk in ("low", "high"):
            risk_keys = sorted({
                key
                for result in seed_results
                for key in risk_block(result, method, risk)
            })
            risk_blocks[risk] = {
                key: mean_se([
                    risk_metric(result, method, risk, key) for result in seed_results
                ])
                for key in risk_keys
            }
        methods[method] = {
            "eval_all": {
                key: mean_se([all_metric(result, method, key) for result in seed_results])
                for key in all_keys
            },
            "eval_by_risk": risk_blocks,
        }

    low_deltas = [
        risk_metric(result, "matched_random", "low", "avg_agent_spread_mean")
        - risk_metric(result, "learned_gate", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    low_delta_stats = mean_se(low_deltas)
    high_deltas = [
        risk_metric(result, "matched_random", "high", "avg_agent_spread_mean")
        - risk_metric(result, "learned_gate", "high", "avg_agent_spread_mean")
        for result in seed_results
    ]
    base_minus_learned = [
        risk_metric(result, "base_no_intervention", "low", "avg_agent_spread_mean")
        - risk_metric(result, "learned_gate", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    learned_minus_oracle = [
        risk_metric(result, "learned_gate", "low", "avg_agent_spread_mean")
        - risk_metric(result, "oracle_gate", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    no_low = [
        risk_metric(result, "base_no_intervention", "low", "avg_agent_spread_mean")
        for result in seed_results
    ]
    response_gaps = [
        float(object_mapping(
            result["counterfactual_response_diagnostic"],
            "counterfactual_response_diagnostic",
        )[
            "low_risk_high_spread_expected_half_spread_reduction"
        ])
        for result in seed_results
    ]
    aggregate_counts = {
        key: sum(
            int(object_mapping(
                result["learned_gate_counts"],
                "learned_gate_counts",
            )[key])
            for result in seed_results
        )
        for key in (
            "n_steps",
            "active",
            "target_positive",
            "true_positive",
            "false_positive",
            "high_risk_steps",
            "high_risk_active",
        )
    }
    gate_rates = rates_from_counts(aggregate_counts)
    learned_alpha = float(np.mean([
        all_metric(result, "learned_gate", "routing_effective_alpha_mean")
        for result in seed_results
    ]))
    matched_alpha = float(np.mean([
        all_metric(result, "matched_random", "routing_effective_alpha_mean")
        for result in seed_results
    ]))
    learned_rewards = [
        all_metric(result, "learned_gate", "mean_reward_mean")
        for result in seed_results
    ]
    matched_rewards = [
        all_metric(result, "matched_random", "mean_reward_mean")
        for result in seed_results
    ]
    include_full_main = "circular_shuffled" in method_names
    circular_deltas: list[float] = []
    no_state_pool_deltas: list[float] = []
    no_routing_improvements: list[float] = []
    if include_full_main:
        circular_deltas = [
            risk_metric(
                result,
                "circular_shuffled",
                "low",
                "avg_agent_spread_mean",
            )
            - risk_metric(
                result,
                "learned_gate",
                "low",
                "avg_agent_spread_mean",
            )
            for result in seed_results
        ]
        no_state_pool_deltas = [
            risk_metric(
                result,
                "no_state_pool_matched",
                "low",
                "avg_agent_spread_mean",
            )
            - risk_metric(
                result,
                "no_state_pool_learned",
                "low",
                "avg_agent_spread_mean",
            )
            for result in seed_results
        ]
        no_routing_improvements = [
            risk_metric(
                result,
                "base_no_intervention",
                "low",
                "avg_agent_spread_mean",
            )
            - risk_metric(
                result,
                "no_routing_trained",
                "low",
                "avg_agent_spread_mean",
            )
            for result in seed_results
        ]
    checks = {
        "gate_train_validation_and_eval_seeds_disjoint": provenance_disjoint,
        "training_schedule_exact": bool(all(
            bool(result["training_schedule_exact"]) for result in seed_results
        )),
        "eval_total_and_risk_stratified_dose_exact": bool(all(
            bool(result["eval_risk_stratified_dose_exact"]) for result in seed_results
        )),
        "gate_off_adapter_exactly_matches_base": bool(all(
            bool(result["gate_off_base_equivalent"]) for result in seed_results
        )),
        "no_intervention_failure_rate_at_least_70pct": bool(
            np.mean(np.asarray(no_low) >= 4.0) >= 0.70
        ),
        "learned_activation_in_registered_range": bool(
            min_activation <= gate_rates["activation_rate"] <= max_activation
        ),
        "learned_high_risk_fpr_below_limit": bool(
            gate_rates["high_risk_fpr"] <= max_high_risk_fpr
        ),
        "learned_target_recall_above_minimum": bool(
            gate_rates["recall"] >= min_target_recall
        ),
        "learned_contract_allocates_more_effective_pool_than_matched": bool(
            learned_alpha > matched_alpha
        ),
        "counterfactual_expected_response_is_material": bool(
            float(np.mean(response_gaps)) >= min_expected_response
        ),
        "counterfactual_response_positive_in_at_least_70pct_seeds": bool(
            np.mean(np.asarray(response_gaps) > 0.0) >= min_positive_seed_rate
        ),
        "learned_beats_matched_on_low_risk_spread": bool(
            float(np.mean(low_deltas)) > 0.0
        ),
        "paired_positive_seed_rate_at_least_70pct": bool(
            np.mean(np.asarray(low_deltas) > 0.0) >= min_positive_seed_rate
        ),
        "paired_normal_approx_95pct_ci_lower_above_zero": bool(
            float(low_delta_stats["mean"])
            - 1.96 * float(low_delta_stats["se"])
            > 0.0
        ),
        "learned_improves_base_low_risk_spread": bool(
            float(np.mean(base_minus_learned)) > 0.0
        ),
        "high_risk_spread_not_reduced_more_than_half_tick": bool(
            float(np.mean(high_deltas)) <= 0.5
        ),
        "reward_not_collapsed_vs_matched": bool(
            float(np.mean(learned_rewards)) >= float(np.mean(matched_rewards)) - 0.005
        ),
    }
    if include_full_main:
        checks.update({
            "eval_circular_risk_stratified_dose_exact": bool(all(
                bool(result["eval_circular_risk_stratified_dose_exact"])
                for result in seed_results
            )),
            "ablation_gate_off_adapters_exactly_match_base": bool(all(
                bool(result["ablation_gate_off_base_equivalent"])
                for result in seed_results
            )),
            "learned_beats_circular_shuffle_on_low_risk_spread": bool(
                float(np.mean(circular_deltas)) > 0.0
            ),
            "circular_paired_positive_seed_rate_at_least_70pct": bool(
                np.mean(np.asarray(circular_deltas) > 0.0)
                >= min_positive_seed_rate
            ),
            "state_dependent_pool_strengthens_timing_advantage": bool(
                float(np.mean(low_deltas))
                > float(np.mean(no_state_pool_deltas))
            ),
            "routing_training_strengthens_learned_improvement": bool(
                float(np.mean(base_minus_learned))
                > float(np.mean(no_routing_improvements))
            ),
        })
    output: Dict[str, object] = {
        "methods": methods,
        "learned_gate_counts": aggregate_counts,
        "learned_gate_rates": gate_rates,
        "base_no_intervention_low_risk_spreads": no_low,
        "paired_matched_minus_learned_low_risk_spread": {
            **low_delta_stats,
            "positive_seed_count": int(np.sum(np.asarray(low_deltas) > 0.0)),
            "n_seeds": len(low_deltas),
            "values": low_deltas,
        },
        "paired_matched_minus_learned_high_risk_spread": {
            **mean_se(high_deltas),
            "values": high_deltas,
        },
        "paired_base_minus_learned_low_risk_spread": {
            **mean_se(base_minus_learned),
            "values": base_minus_learned,
        },
        "paired_learned_minus_oracle_low_risk_spread": {
            **mean_se(learned_minus_oracle),
            "values": learned_minus_oracle,
        },
        "counterfactual_signal_response": {
            **mean_se(response_gaps),
            "positive_seed_count": int(np.sum(np.asarray(response_gaps) > 0.0)),
            "n_seeds": len(response_gaps),
            "values": response_gaps,
        },
        "learned_effective_alpha": learned_alpha,
        "matched_effective_alpha": matched_alpha,
        "oracle_activation_rate": float(np.mean([
            float(cast(Any, result["oracle_activation_rate"]))
            for result in seed_results
        ])),
        "pass_checks": checks,
        "passes_all": bool(all(checks.values())),
    }
    if include_full_main:
        output["full_main_diagnostics"] = {
            "paired_circular_minus_learned_low_risk_spread": {
                **mean_se(circular_deltas),
                "positive_seed_count": int(np.sum(
                    np.asarray(circular_deltas) > 0.0
                )),
                "n_seeds": len(circular_deltas),
                "values": circular_deltas,
            },
            "paired_no_state_pool_matched_minus_learned_low_risk_spread": {
                **mean_se(no_state_pool_deltas),
                "positive_seed_count": int(np.sum(
                    np.asarray(no_state_pool_deltas) > 0.0
                )),
                "n_seeds": len(no_state_pool_deltas),
                "values": no_state_pool_deltas,
            },
            "paired_base_minus_no_routing_trained_low_risk_spread": {
                **mean_se(no_routing_improvements),
                "positive_seed_count": int(np.sum(
                    np.asarray(no_routing_improvements) > 0.0
                )),
                "n_seeds": len(no_routing_improvements),
                "values": no_routing_improvements,
            },
        }
    return output


def validate_gate_artifact(
    payload: Dict[str, object],
    eval_seeds: Sequence[int],
    config: LocalLOBConfig,
) -> Dict[str, object]:
    metadata = object_mapping(payload.get("metadata", {}), "metadata")
    train_seed_values = metadata.get("train_seeds", [])
    validation_seed_values = metadata.get("validation_seeds", [])
    if not isinstance(train_seed_values, list) or not isinstance(
        validation_seed_values,
        list,
    ):
        raise ValueError("gate train and validation seeds must be JSON lists")
    train_seeds = [int(value) for value in train_seed_values]
    validation_seeds = [int(value) for value in validation_seed_values]
    used = set(train_seeds) | set(validation_seeds)
    overlap = sorted(used & set(eval_seeds))
    if overlap:
        raise ValueError(f"gate train/validation seeds overlap evaluation seeds: {overlap}")
    if metadata.get("uses_intervention_outcomes") is not False:
        raise ValueError("learned gate must be trained without intervention outcomes")
    if metadata.get("label_definition") != "low_risk AND calibrated_contract_severity_gt_zero":
        raise ValueError("unexpected learned-gate label definition")
    artifact_config = object_mapping(
        metadata.get("environment_config", {}),
        "environment_config",
    )
    normalized_artifact_config = json.loads(json.dumps(dict(artifact_config)))
    runtime_config = json.loads(json.dumps(asdict(config)))
    for key, runtime_value in runtime_config.items():
        if key != "seed" and normalized_artifact_config.get(key) != runtime_value:
            raise ValueError(f"learned-gate artifact/config mismatch for {key}")
    validation_metrics = object_mapping(
        payload.get("validation_metrics", {}),
        "validation_metrics",
    )
    return {
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "evaluation_seeds": list(eval_seeds),
        "disjoint": not overlap,
        "passive_training_only": True,
        "environment_config_matches": True,
        "validation_metrics": dict(validation_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("engineering", "screen", "heldout", "full_main"),
        default="screen",
    )
    parser.add_argument(
        "--full-main-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run circular, aggressive-control, and trained-ablation arms",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed per-seed files with the same protocol fingerprint",
    )
    parser.add_argument("--learned-gate-artifact", required=True)
    parser.add_argument("--equilibrium-episodes", type=int, default=100)
    parser.add_argument("--response-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seeds", default="45-49")
    parser.add_argument("--response-train-rate", type=float, default=1.0)
    parser.add_argument("--calibration-episodes", type=int, default=5)
    parser.add_argument("--calibration-quantile", type=float, default=0.75)
    parser.add_argument("--calibration-upper-quantile", type=float, default=0.90)
    parser.add_argument("--min-expected-response", type=float, default=0.05)
    parser.add_argument("--min-positive-seed-rate", type=float, default=0.70)
    parser.add_argument("--min-activation", type=float, default=0.05)
    parser.add_argument("--max-activation", type=float, default=0.40)
    parser.add_argument("--max-high-risk-fpr", type=float, default=0.05)
    parser.add_argument("--min-target-recall", type=float, default=0.70)
    parser.add_argument("--routing-rho", type=float, default=1.0)
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
        default="revision/experiments/outputs/lob_mappo_learned_gate_screen_v1",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    device = choose_device(args.device)
    config = make_environment_config(seeds[0] if seeds else 0, args.episode_length)
    artifact_path = Path(args.learned_gate_artifact)
    artifact_bytes = artifact_path.read_bytes()
    payload: Dict[str, object] = json.loads(artifact_bytes.decode("utf-8"))
    provenance = validate_gate_artifact(payload, seeds, config)
    validation_metrics = object_mapping(
        provenance["validation_metrics"],
        "validation_metrics",
    )
    if not (
        args.min_activation <= float(validation_metrics.get("activation_rate", -1.0)) <= 0.35
        and float(validation_metrics.get("high_risk_fpr", 1.0)) <= args.max_high_risk_fpr
    ):
        raise ValueError("frozen learned gate did not pass its validation constraints")
    model = LearnedLOBGateModel.from_dict(payload)
    gate_config = load_lob_gate_config(args.gate_config)
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

    include_full_main = bool(args.stage == "full_main" or args.full_main_controls)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    protocol_identity = {
        "stage": args.stage,
        "seeds": seeds,
        "include_full_main_controls": include_full_main,
        "equilibrium_episodes": args.equilibrium_episodes,
        "response_episodes": args.response_episodes,
        "eval_episodes": args.eval_episodes,
        "response_train_rate": args.response_train_rate,
        "calibration_episodes": args.calibration_episodes,
        "calibration_quantile": args.calibration_quantile,
        "calibration_upper_quantile": args.calibration_upper_quantile,
        "min_expected_response": args.min_expected_response,
        "min_positive_seed_rate": args.min_positive_seed_rate,
        "min_activation": args.min_activation,
        "max_activation": args.max_activation,
        "max_high_risk_fpr": args.max_high_risk_fpr,
        "min_target_recall": args.min_target_recall,
        "routing_rho": args.routing_rho,
        "warm_start_steps": args.warm_start_steps,
        "warm_start_updates": args.warm_start_updates,
        "warm_start_low_half_spread": args.warm_start_low_half_spread,
        "warm_start_high_half_spread": args.warm_start_high_half_spread,
        "gate_artifact_sha256": artifact_sha256,
        "config": asdict(config),
        "gate_config": asdict(gate_config),
        "mappo_config": asdict(mappo),
    }
    protocol_fingerprint = hashlib.sha256(json.dumps(
        protocol_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = output_dir / "seed_results"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_fingerprint") != protocol_fingerprint:
            raise ValueError("resume manifest does not match the current protocol")
    else:
        manifest_path.write_text(json.dumps({
            "protocol_fingerprint": protocol_fingerprint,
            "protocol_identity": protocol_identity,
        }, indent=2), encoding="utf-8")

    seed_results: List[Dict[str, object]] = []
    for seed in seeds:
        seed_path = seed_dir / f"seed_{seed}.json"
        if args.resume and seed_path.exists():
            envelope = json.loads(seed_path.read_text(encoding="utf-8"))
            if envelope.get("protocol_fingerprint") != protocol_fingerprint:
                raise ValueError(f"seed {seed} result has a different protocol")
            print(f"seed={seed} resume-complete", flush=True)
            seed_results.append(envelope["result"])
            continue
        print(f"seed={seed} learned-gate-vs-matched-random", flush=True)
        result = run_seed(
            replace(config, seed=seed),
            mappo,
            gate_config,
            model,
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
            include_full_main=include_full_main,
        )
        seed_results.append(result)
        temporary_path = seed_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps({
            "protocol_fingerprint": protocol_fingerprint,
            "result": result,
        }, indent=2), encoding="utf-8")
        temporary_path.replace(seed_path)

    stage_name = (
        "local_lob_mappo_learned_gate_full_main"
        if args.stage == "full_main"
        else f"local_lob_mappo_learned_gate_{args.stage}_not_full_main"
    )
    summary: Dict[str, object] = {
        "metadata": {
            "stage": stage_name,
            "device": device,
            "seeds": seeds,
            "protocol_fingerprint": protocol_fingerprint,
            "per_seed_resume_enabled": args.resume,
            "full_main_controls_and_ablations": include_full_main,
            "equilibrium_episodes": args.equilibrium_episodes,
            "response_episodes": args.response_episodes,
            "eval_episodes": args.eval_episodes,
            "response_train_rate": args.response_train_rate,
            "calibration_episodes": args.calibration_episodes,
            "calibration_quantile": args.calibration_quantile,
            "calibration_upper_quantile": args.calibration_upper_quantile,
            "gate_artifact": str(artifact_path),
            "gate_artifact_sha256": artifact_sha256,
            "gate_provenance": provenance,
            "primary_comparator": "within_episode_risk_stratified_exact_dose_permutation",
            "schedule_source": "passive_base_policy_crn_reference_path",
            "same_frozen_response_policy_all_intervention_arms": True,
            "same_action_uniforms_all_arms": True,
            "oracle_is_diagnostic_upper_bound_only": True,
            "acceptance_thresholds": {
                "min_expected_response": args.min_expected_response,
                "min_positive_seed_rate": args.min_positive_seed_rate,
                "min_activation": args.min_activation,
                "max_activation": args.max_activation,
                "max_high_risk_fpr": args.max_high_risk_fpr,
                "min_target_recall": args.min_target_recall,
            },
            "config": asdict(config),
            "gate_config": asdict(gate_config),
            "mappo_config": asdict(mappo),
        },
        "seed_results": seed_results,
        "aggregate": aggregate(
            seed_results,
            bool(provenance["disjoint"]),
            args.min_expected_response,
            args.min_positive_seed_rate,
            args.min_activation,
            args.max_activation,
            args.max_high_risk_fpr,
            args.min_target_recall,
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    aggregate_block = object_mapping(summary["aggregate"], "aggregate")
    print(json.dumps({
        "output_dir": str(output_dir),
        "paired_low_risk": aggregate_block[
            "paired_matched_minus_learned_low_risk_spread"
        ],
        "learned_gate_rates": aggregate_block["learned_gate_rates"],
        "pass_checks": aggregate_block["pass_checks"],
        "passes_all": aggregate_block["passes_all"],
    }, indent=2))


if __name__ == "__main__":
    main()
