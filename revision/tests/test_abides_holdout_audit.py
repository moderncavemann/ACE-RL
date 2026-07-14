from __future__ import annotations

import json
import sys
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from run_abides_holdout_audit_evaluation import (  # noqa: E402
    CELL_CONFIGS,
    EVALUATOR_VERSION,
    evaluation_fingerprint,
    normalize_compact_record,
    task_paths,
)


def test_holdout_audit_fingerprint_is_order_sensitive_and_stable() -> None:
    left = [{"cell": "e1", "sha": "a"}, {"cell": "e2", "sha": "b"}]
    assert evaluation_fingerprint(left) == evaluation_fingerprint(left)
    assert evaluation_fingerprint(left) != evaluation_fingerprint(list(reversed(left)))
    assert EVALUATOR_VERSION == "abides_holdout_audit_evaluation_v2"


def test_holdout_audit_task_paths_are_cell_and_seed_scoped() -> None:
    e1_result, e1_steps = task_paths("ecology_e1", 560000)
    e2_result, _ = task_paths("ecology_e2", 570000)
    assert e1_result.name == "result.json"
    assert e1_steps.name == "step_records.jsonl.gz"
    assert e1_result != e2_result


def test_ecology_audit_configs_are_frozen_and_disjoint() -> None:
    configs = {
        cell: json.loads(path.read_text(encoding="utf-8"))
        for cell, path in CELL_CONFIGS.items()
    }
    assert set(configs) == {"ecology_e1", "ecology_e2"}
    assert set(configs["ecology_e1"]["policy_seeds"]).isdisjoint(
        configs["ecology_e2"]["policy_seeds"]
    )
    for config in configs.values():
        assert config["response_episodes"] == 96
        assert len(config["policy_seeds"]) == 10
        assert len(config["evaluation_seeds"]) == 10
        assert len(config["methods"]) == 6


def test_compact_record_normalization_preserves_flow_and_derives_tv() -> None:
    record = {
        "info": {
            "flow_shares": [0.1, 0.2, 0.3, 0.4],
            "routing_effective_alpha": 0.25,
            "routing_severity": 0.5,
        }
    }
    normalized = normalize_compact_record(record)
    info = normalized["info"]
    assert info["routing_weights"] == record["info"]["flow_shares"]
    assert info["routing_total_variation"] == 0.2
    assert info["routing_mechanism_parameters"]["audit_weights_source"] == (
        "realized_flow_shares"
    )
    assert "routing_weights" not in record["info"]
