from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO = REVISION_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "experiments"))

from experiments.run_abides_abce_primary_followup import (  # noqa: E402
    METHODS,
    disallowed_worktree_entries,
    pending_policy_seeds,
    source_bundle_sha256,
)
from experiments.run_abides_abce_pipeline import (  # noqa: E402
    current_git_commit,
    validate_official_smoke,
    validate_smoke,
)
from src.agents.abides_mappo import rollout_episode  # noqa: E402
from src.interventions.abides_abce import (  # noqa: E402
    circular_schedule,
    exact_random_schedule,
    frozen_identity_permutation,
    lagged_dealer_schedule,
    periodic_schedule,
)


def load_config(name: str) -> dict:
    return json.loads((REVISION_ROOT / "configs" / name).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_abce_schedule_controls_preserve_the_exact_continuous_multiset() -> None:
    source = [0.0, 0.72, 0.0, 0.83, 0.91, 0.0, 0.77, 0.0]
    random = exact_random_schedule(source, 1201)
    circular = circular_schedule(source, 1202)
    periodic = periodic_schedule(source)
    assert random.exact_multiset_of(source)
    assert circular.exact_multiset_of(source)
    assert periodic.exact_multiset_of(source)
    assert list(random.values) != source
    assert list(circular.values) != source
    active = np.flatnonzero(np.asarray(periodic.values) > 0.0)
    assert max(np.diff(active)) - min(np.diff(active)) <= 1


def test_abce_lagged_revelation_and_identity_permutation_are_exact() -> None:
    applied = [0.1, 0.2, 0.0, 0.8]
    assert lagged_dealer_schedule(applied) == (0.0, 0.1, 0.2, 0.0)
    left = frozen_identity_permutation(10, 534000, 12, 541317)
    right = frozen_identity_permutation(10, 534000, 12, 541317)
    assert left == right
    assert sorted(left) == list(range(10))
    assert left != tuple(range(10))


class FakeTrainer:
    def __init__(self):
        self.actor_inputs = []

    def act(self, obs, critic, deterministic=False, action_uniforms=None):
        del critic, deterministic, action_uniforms
        self.actor_inputs.append(np.asarray(obs, dtype=float).copy())
        count = obs.shape[0]
        return (
            np.zeros(count, dtype=np.int64),
            np.zeros(count, dtype=float),
            np.zeros(count, dtype=float),
        )


class FakeEnv:
    def __init__(self, steps: int):
        self.steps = steps
        self.current = 0
        self.applied = []

    def state(self):
        return {
            "backend": "abides",
            "observations": [[0.0] * 6 for _ in range(10)],
            "rewards": [0.0] * 10,
            "done": self.current >= self.steps,
            "agents": [{"inventory": 0} for _ in range(10)],
            "info": {
                "avg_half_spread": 1.0,
                "public_volatility": 0.0,
                "completed_steps": self.current,
            },
        }

    def reset(self, seed: int):
        del seed
        self.current = 0
        return self.state()

    def step(self, actions, **kwargs):
        del actions
        self.applied.append(float(kwargs["routing_strength"]))
        self.current += 1
        return self.state()

    def close_episode(self):
        return None


def test_rollout_separates_platform_applied_and_dealer_observed_signal() -> None:
    env = FakeEnv(3)
    trainer = FakeTrainer()
    result = rollout_episode(
        env,
        trainer,
        seed=1,
        episode_steps=3,
        train=False,
        announced_signal_schedule=[0.2, 0.4, 0.8],
        dealer_signal_schedule=[0.0, 0.2, 0.4],
    )
    assert env.applied == [0.2, 0.4, 0.8]
    assert [row["platform_applied_signal"] for row in result["records"]] == [
        0.2,
        0.4,
        0.8,
    ]
    assert [row["dealer_observed_signal"] for row in result["records"]] == [
        0.0,
        0.2,
        0.4,
    ]
    assert np.allclose(
        [float(obs[0, -1]) for obs in trainer.actor_inputs],
        [0.0, 0.2, 0.4],
        atol=1e-7,
        rtol=0.0,
    )


def test_formal_abce_config_freezes_new_arms_and_seed_namespaces() -> None:
    config = load_config("abides_abce_primary_followup_v1.json")
    assert tuple(config["methods"]) == METHODS
    assert tuple(config["trained_methods"]) == METHODS
    assert config["reuse_existing_controls"] is False
    assert config["policy_seeds"] == list(range(530000, 530020))
    assert config["evaluation_seeds"] == list(range(534000, 534010))
    assert config["response_episodes"] == 96
    assert config["minimum_converged_policy_seeds"] == 18
    assert config["adapter_source_bundle_sha256"] == source_bundle_sha256()


def test_pending_policy_seeds_are_unique_and_resume_completed_work() -> None:
    assert pending_policy_seeds([1, 2, 3], [{"policy_seed": 2}]) == [1, 3]
    with pytest.raises(ValueError, match="duplicates"):
        pending_policy_seeds([1, 1], [])


def test_formal_worktree_allows_only_managed_untracked_artifact() -> None:
    root = "revision/paper/artifacts/abides_abce_primary_followup_v1"
    assert disallowed_worktree_entries([f"?? {root}/checks.json"], root) == []
    assert disallowed_worktree_entries(["?? revision/new_runner.py"], root) == [
        "?? revision/new_runner.py"
    ]
    assert disallowed_worktree_entries([f" M {root}/checks.json"], root) == [
        f" M {root}/checks.json"
    ]


def test_exact_official_holdout_uses_risk_only_signal_and_frozen_selection() -> None:
    exact = load_config("abides_standard_learner_exact_risk_only_holdout_v1.json")
    original = load_config("abides_standard_learner_holdout_v1.json")
    assert exact["include_lagged_spread_severity"] is False
    assert exact["selected_hyperparameters"] == original["selected_hyperparameters"]
    assert exact["selection_artifact_sha256"] == original[
        "selection_artifact_sha256"
    ]
    assert exact["policy_seeds"] == list(range(560000, 560010))
    assert exact["response_evaluation_seeds"] == list(range(562000, 562010))
    assert exact["methods"][-1] == "brero_threshold_buybox_adaptation"
    assert exact["brero_parameters"]["budget_match_status"] == (
        "unmatched_descriptive_only"
    )
    assert exact["protocol_file_sha256"] == file_sha256(
        REPO / exact["protocol_file"]
    )
    assert exact["execution_amendment_file_sha256"] == file_sha256(
        REPO / exact["execution_amendment_file"]
    )


def test_pipeline_smoke_gate_accepts_integer_convergence_count(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    config = {
        "output_dir": str(output),
        "policy_seeds": [1, 2],
        "minimum_converged_policy_seeds": 2,
    }
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"git_commit": current_git_commit()}), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "implementation_status": "pass",
                "checks": {"grid": True},
                "converged_policy_seeds": 2,
                "classifications": {
                    "A": "smoke_only",
                    "B": "smoke_only",
                    "C1": "smoke_only",
                    "C2": "smoke_only",
                },
            }
        ),
        encoding="utf-8",
    )
    for seed in config["policy_seeds"]:
        path = output / "policy_seeds" / f"seed_{seed}" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    receipt = validate_smoke(config_path)
    assert receipt["converged_policy_seed_count"] == 2


def test_pipeline_requires_current_commit_official_risk_only_smoke(
    tmp_path: Path,
) -> None:
    base = tmp_path / "official"
    output = Path(f"{base}_formal_smoke_v2")
    config = {
        "output_dir": str(base),
        "learners": ["official_full_policy_MAPPO", "official_IPPO"],
        "policy_seeds": [560000],
    }
    manifest = {
        "git_commit": current_git_commit(),
        "stage": "formal",
        "engineering_smoke": True,
        "config": {"include_lagged_spread_severity": False},
    }
    (output / "formal").mkdir(parents=True)
    (output / "formal" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "implementation_status": "pass",
                "checks": {"task_count": True},
                "learner_results": {learner: {} for learner in config["learners"]},
            }
        ),
        encoding="utf-8",
    )
    for learner in config["learners"]:
        path = output / "formal" / learner / "seed_560000" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    receipt = validate_official_smoke(config)
    assert receipt["task_count"] == 2
