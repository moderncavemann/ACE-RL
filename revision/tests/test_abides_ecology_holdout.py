from __future__ import annotations

import json
import sys
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from run_abides_mappo_full_main import protocol_fingerprint  # noqa: E402


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_ecology_cells_expand_preregistered_seed_and_market_contracts() -> None:
    parent = load_config("abides_ecology_holdout_v1.json")
    pairs = (
        (
            "E1_fast_transition",
            "abides_ecology_e1_base_v1.json",
            "abides_ecology_e1_response_v1.json",
        ),
        (
            "E2_weaker_informed_flow",
            "abides_ecology_e2_base_v1.json",
            "abides_ecology_e2_response_v1.json",
        ),
    )
    for cell_name, base_name, response_name in pairs:
        cell = parent["cells"][cell_name]
        base = load_config(base_name)
        response = load_config(response_name)
        expected_market = parent["common_market"] | cell["market_overrides"]
        assert base["base_training_seeds"] == cell["policy_seeds"]
        assert base["base_fixed_evaluation_seeds"] == cell["base_evaluation_seeds"]
        assert base["reserved_response_evaluation_seeds"] == cell[
            "response_evaluation_seeds"
        ]
        assert response["policy_seeds"] == cell["policy_seeds"]
        assert response["evaluation_seeds"] == cell["response_evaluation_seeds"]
        assert base["market"] == expected_market
        assert response["market"] == expected_market
        assert response["base_protocol_fingerprint"] == protocol_fingerprint(base)


def test_ecology_cells_use_fresh_disjoint_namespaces_and_fixed_budgets() -> None:
    base_names = (
        "abides_ecology_e1_base_v1.json",
        "abides_ecology_e2_base_v1.json",
    )
    response_names = (
        "abides_ecology_e1_response_v1.json",
        "abides_ecology_e2_response_v1.json",
    )
    all_seeds: list[set[int]] = []
    namespaces: set[int] = set()
    for base_name, response_name in zip(base_names, response_names, strict=True):
        base = load_config(base_name)
        response = load_config(response_name)
        assert base["base_training_mode"] == "fixed_horizon"
        assert base["base_max_episodes"] == 384
        assert response["response_episodes"] == 96
        assert response["required_policy_seed_count"] == 10
        assert set(response["methods"]) >= {
            "no_intervention",
            "risk_aware",
            "matched_random",
            "no_routing",
        }
        all_seeds.extend(
            (
                set(base["base_training_seeds"]),
                set(base["base_fixed_evaluation_seeds"]),
                set(response["evaluation_seeds"]),
            )
        )
        for key in (
            "base_environment_seed_namespace",
            "base_action_namespace",
            "base_evaluation_action_namespace",
        ):
            assert int(base[key]) not in namespaces
            namespaces.add(int(base[key]))
        for key in (
            "response_environment_seed_namespace",
            "response_action_namespace",
            "evaluation_action_namespace",
        ):
            assert int(response[key]) not in namespaces
            namespaces.add(int(response[key]))
    for index, left in enumerate(all_seeds):
        for right in all_seeds[index + 1 :]:
            assert left.isdisjoint(right)
