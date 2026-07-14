from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from run_abides_dose_routing_sensitivity import (  # noqa: E402
    engineering_smoke_config,
    group_specs,
    routing_classification,
)


def load_config() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "abides_dose_routing_sensitivity_v1.json"
        ).read_text()
    )


def test_formal_design_has_three_dose_and_five_routing_groups() -> None:
    specs = group_specs(load_config())
    assert len(specs) == 8
    assert sum(spec["study"] == "dose" for spec in specs) == 3
    assert sum(spec["study"] == "routing" for spec in specs) == 5
    assert len({spec["group_id"] for spec in specs}) == 8


def test_engineering_smoke_is_isolated_from_formal_namespaces() -> None:
    formal = load_config()
    smoke = engineering_smoke_config(formal)
    assert smoke["engineering_smoke"]
    assert len(group_specs(smoke)) == 2
    assert set(smoke["evaluation_seeds"]).isdisjoint(formal["evaluation_seeds"])
    assert smoke["output_dir"] != formal["output_dir"]


def test_routing_classification_does_not_treat_nonsignificance_as_equivalence() -> None:
    specs = [spec for spec in group_specs(load_config()) if spec["study"] == "routing"]
    statistics = {}
    for index, spec in enumerate(specs):
        statistics[spec["group_id"]] = {
            "matched_random_minus_risk_aware_low_spread": {
                "policy_seed_t_interval": {
                    "mean": 0.1 if index < 3 else -0.01,
                    "ci_low": -0.05,
                    "ci_high": 0.15,
                }
            }
        }
    result = routing_classification(specs, statistics)
    assert result["classification"] == "broadly_robust"
    assert result["nonsignificance_is_not_equivalence"]
