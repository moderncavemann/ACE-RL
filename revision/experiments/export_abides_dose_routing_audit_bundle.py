#!/usr/bin/env python3
"""Export the frozen ABIDES dose and routing study as an audit bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_abides_dose_routing_sensitivity import group_specs  # noqa: E402
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from src.utils.abides_audit import (  # noqa: E402
    METRIC_FIELDS,
    exact_sign_test,
    holm_adjust,
    t_interval,
    two_way_paired_bootstrap,
)


DEFAULT_CONFIG = "revision/configs/abides_dose_routing_sensitivity_v1.json"
DEFAULT_OUTPUT = "revision/experiments/outputs/abides_dose_routing_sensitivity_v1"
DEFAULT_BUNDLE = "revision/paper/artifacts/abides_dose_routing_sensitivity_v1"
PAPER_PER_PATH = "revision/paper/data/abides_dose_routing_sensitivity_v1_per_path.csv"
PAPER_PER_SEED = "revision/paper/data/abides_dose_routing_sensitivity_v1_per_seed.csv"
PAPER_DOSE_FIGURE = "revision/paper/figures/abides_dose_curve.png"
PAPER_ROUTING_FIGURE = "revision/paper/figures/abides_routing_sensitivity.png"
ABIDES_COMMIT = "f9cbe51342b7dedd9587e4e069040d68a5c6477f"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_csv(source: Path, target: Path) -> None:
    rows = read_csv(source)
    if not rows:
        raise RuntimeError(f"CSV has no rows: {source}")
    fieldnames = list(rows[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def validate_archival_commit(commit: str | None) -> str | None:
    if commit is None:
        return None
    full = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=REPO, text=True
    ).strip()
    if len(full) != 40:
        raise RuntimeError("result archival commit did not resolve to a full SHA")
    return full


def load_task_results(
    output_dir: Path, specs: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    results = []
    for spec in specs:
        for policy_seed in seeds:
            path = (
                output_dir
                / "groups"
                / str(spec["group_id"])
                / f"seed_{int(policy_seed)}"
                / "result.json"
            )
            if not path.exists():
                raise RuntimeError(f"missing sensitivity task result: {path}")
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def seed_metric_values(
    rows: Sequence[Mapping[str, str]], group_id: str, method: str, metric: str
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["group_id"] == group_id and row["method"] == method:
            grouped[int(row["policy_seed"])].append(float(row[metric]))
    if not grouped:
        raise RuntimeError(f"no rows for {group_id}/{method}/{metric}")
    return [float(mean(grouped[seed])) for seed in sorted(grouped)]


def group_statistics(
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for spec in specs:
        group_id = str(spec["group_id"])
        methods = ["no_intervention", *list(spec["methods"])]
        method_stats: dict[str, Any] = {}
        selected_metrics = (
            "low_execution_half_spread",
            "all_state_execution_half_spread",
            "dealer_reward_units_per_episode",
            "activation_rate",
            "high_risk_activation_rate",
            "mean_routing_total_variation",
            "mean_volume",
        )
        for method in methods:
            method_stats[method] = {
                metric: t_interval(
                    seed_metric_values(rows, group_id, method, metric)
                )
                for metric in selected_metrics
            }
        no_reward = seed_metric_values(
            rows, group_id, "no_intervention", "dealer_reward_units_per_episode"
        )
        for method in methods:
            method_reward = seed_metric_values(
                rows, group_id, method, "dealer_reward_units_per_episode"
            )
            method_stats[method]["dealer_reward_loss_vs_no_intervention"] = (
                t_interval([base - value for base, value in zip(no_reward, method_reward)])
            )

        contrasts: dict[str, Any] = {}
        group_rows = [row for row in rows if row["group_id"] == group_id]
        risk_spread = seed_metric_values(
            rows, group_id, "risk_aware", "low_execution_half_spread"
        )
        for comparator in spec["methods"]:
            if comparator == "risk_aware":
                continue
            comparator_spread = seed_metric_values(
                rows, group_id, str(comparator), "low_execution_half_spread"
            )
            differences = [
                left - right for left, right in zip(comparator_spread, risk_spread)
            ]
            interval = t_interval(differences)
            contrast_id = f"{group_id}:{comparator}"
            contrasts[str(comparator)] = {
                "policy_seed_t_interval": interval,
                "policy_seed_exact_sign_test": exact_sign_test(differences),
                "two_way_paired_bootstrap": two_way_paired_bootstrap(
                    group_rows,
                    str(comparator),
                    int(config["bootstrap_replicates"]),
                    int(config["bootstrap_seed"]),
                ),
            }
            raw_p_values[contrast_id] = float(interval["p_value_two_sided"])

        risk_activation = float(method_stats["risk_aware"]["activation_rate"]["mean"])
        risk_tv = float(
            method_stats["risk_aware"]["mean_routing_total_variation"]["mean"]
        )
        budget = {}
        for comparator in spec["methods"]:
            if comparator == "risk_aware":
                continue
            comparator_activation = float(
                method_stats[str(comparator)]["activation_rate"]["mean"]
            )
            comparator_tv = float(
                method_stats[str(comparator)]["mean_routing_total_variation"]["mean"]
            )
            tv_ratio = comparator_tv / risk_tv if risk_tv > 1e-15 else math.nan
            budget[str(comparator)] = {
                "activation_absolute_difference": abs(
                    comparator_activation - risk_activation
                ),
                "routing_tv_ratio_comparator_over_risk_aware": tv_ratio,
                "passive_calibration_tolerances": {
                    "activation_difference_maximum": float(
                        config["calibration"]["maximum_pair_activation_difference"]
                    ),
                    "routing_tv_ratio_minimum": float(
                        config["calibration"]["minimum_tv_ratio"]
                    ),
                    "routing_tv_ratio_maximum": float(
                        config["calibration"]["maximum_tv_ratio"]
                    ),
                },
                "formal_realized_budget_within_passive_tolerances": bool(
                    abs(comparator_activation - risk_activation)
                    <= float(
                        config["calibration"]["maximum_pair_activation_difference"]
                    )
                    and math.isfinite(tv_ratio)
                    and float(config["calibration"]["minimum_tv_ratio"])
                    <= tv_ratio
                    <= float(config["calibration"]["maximum_tv_ratio"])
                ),
            }
        output[group_id] = {
            "study": spec["study"],
            "dose_target": float(spec["dose_target"]),
            "routing_target_kappa": float(spec["routing_target_kappa"]),
            "routing_routable_fraction": float(spec["routing_routable_fraction"]),
            "method_seed_statistics": method_stats,
            "contrasts": contrasts,
            "formal_realized_budget_diagnostics": budget,
        }

    adjusted = holm_adjust(raw_p_values)
    for contrast_id, adjusted_p in adjusted.items():
        group_id, comparator = contrast_id.split(":", 1)
        output[group_id]["contrasts"][comparator][
            "holm_adjusted_p_value_global_family"
        ] = adjusted_p
    return {
        "groups": output,
        "holm_family": sorted(raw_p_values),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "bootstrap_seed": int(config["bootstrap_seed"]),
    }


def routing_classification(statistics: Mapping[str, Any]) -> dict[str, Any]:
    routing = {
        group_id: values
        for group_id, values in statistics["groups"].items()
        if values["study"] == "routing"
    }
    intervals = {
        group_id: values["contrasts"]["matched_random"][
            "policy_seed_t_interval"
        ]
        for group_id, values in routing.items()
    }
    favorable = sum(float(values["mean"]) > 0.0 for values in intervals.values())
    adverse = sorted(
        group_id
        for group_id, values in intervals.items()
        if float(values["ci_high"]) < 0.0
    )
    center = "routing_k4_f050"
    center_supported = float(intervals[center]["ci_low"]) > 0.0
    if favorable >= 3 and not adverse:
        label = "broadly_robust"
    elif center_supported and not adverse:
        label = "locally_robust"
    else:
        label = "parameter_specific"
    return {
        "classification": label,
        "favorable_mean_settings": favorable,
        "total_settings": len(intervals),
        "significantly_adverse_settings": adverse,
        "center_supported": center_supported,
        "nonsignificance_is_not_equivalence": True,
    }


def combine_step_records(
    output_dir: Path,
    bundle_path: Path,
    specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> int:
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as target:
                for spec in specs:
                    for policy_seed in seeds:
                        source = (
                            output_dir
                            / "groups"
                            / str(spec["group_id"])
                            / f"seed_{int(policy_seed)}"
                            / "evaluation_steps.jsonl.gz"
                        )
                        with gzip.open(source, "rt", encoding="utf-8") as handle:
                            for line in handle:
                                payload = json.loads(line)
                                target.write(canonical_json(payload) + "\n")
                                count += 1
    temporary.replace(bundle_path)
    return count


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "ACE-RL"},
    )
    plt.close(fig)


def render_figures(statistics: Mapping[str, Any], directory: Path) -> None:
    colors = {
        "risk_aware": "#1f6b45",
        "matched_random": "#4b5563",
        "spread_only_budget_matched": "#b45309",
    }
    labels = {
        "risk_aware": "Risk-aware",
        "matched_random": "Matched random",
        "spread_only_budget_matched": "Spread-only",
    }
    dose_groups = sorted(
        (
            values
            for values in statistics["groups"].values()
            if values["study"] == "dose"
        ),
        key=lambda values: float(values["dose_target"]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    panels = (
        ("activation_rate", "Realized activation", "low_execution_half_spread", "Low-risk execution half-spread"),
        ("mean_routing_total_variation", "Mean routing total variation", "low_execution_half_spread", "Low-risk execution half-spread"),
        ("low_execution_half_spread", "Low-risk execution half-spread", "dealer_reward_loss_vs_no_intervention", "Dealer reward loss"),
        ("low_execution_half_spread", "Low-risk execution half-spread", "high_risk_activation_rate", "High-risk activation"),
    )
    for axis, (x_metric, x_label, y_metric, y_label) in zip(axes.flat, panels):
        for method in colors:
            x = [
                float(group["method_seed_statistics"][method][x_metric]["mean"])
                for group in dose_groups
            ]
            y = [
                float(group["method_seed_statistics"][method][y_metric]["mean"])
                for group in dose_groups
            ]
            axis.plot(x, y, marker="o", linewidth=1.8, color=colors[method], label=labels[method])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("ABIDES secondary dose frontier", fontsize=13)
    save_figure(fig, directory / "abides_dose_curve.png")

    routing_groups = sorted(
        (
            (group_id, values)
            for group_id, values in statistics["groups"].items()
            if values["study"] == "routing"
        ),
        key=lambda item: (
            float(item[1]["routing_routable_fraction"]),
            float(item[1]["routing_target_kappa"]),
        ),
    )
    means = []
    lower = []
    upper = []
    tick_labels = []
    for _, group in routing_groups:
        interval = group["contrasts"]["matched_random"]["policy_seed_t_interval"]
        value = float(interval["mean"])
        means.append(value)
        lower.append(value - float(interval["ci_low"]))
        upper.append(float(interval["ci_high"]) - value)
        tick_labels.append(
            f"k={float(group['routing_target_kappa']):g}\nf={float(group['routing_routable_fraction']):.2f}"
        )
    fig, axis = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    x_values = np.arange(len(means))
    axis.errorbar(
        x_values,
        means,
        yerr=np.asarray([lower, upper]),
        fmt="o",
        color="#1f6b45",
        capsize=4,
        linewidth=1.5,
    )
    axis.axhline(0.0, color="#6b7280", linewidth=1.0)
    axis.set_xticks(x_values, tick_labels)
    axis.set_ylabel("Matched random minus risk-aware half-spread")
    axis.set_title("ABIDES routing sensitivity (secondary)")
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    save_figure(fig, directory / "abides_routing_sensitivity.png")


def export_bundle(
    config_path: Path,
    output_dir: Path,
    bundle: Path,
    result_archival_commit: str | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    specs = group_specs(config)
    output_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    aggregate = json.loads((output_dir / "aggregate.json").read_text(encoding="utf-8"))
    completion = json.loads((output_dir / "completion.json").read_text(encoding="utf-8"))
    if aggregate["implementation_status"] != "pass":
        raise RuntimeError("cannot archive a failed sensitivity implementation")
    if not completion["completed_resume_verification"] or not completion[
        "aggregate_unchanged_on_completed_resume"
    ]:
        raise RuntimeError("completed-resume verification is missing")
    results = load_task_results(output_dir, specs, config["source_policy_seeds"])
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("aggregate.json", "completion.json"):
        atomic_write_json(
            bundle / name,
            json.loads((output_dir / name).read_text(encoding="utf-8")),
        )
    normalize_csv(output_dir / "per_path.csv", bundle / "per_path.csv")
    normalize_csv(output_dir / "per_seed.csv", bundle / "per_seed.csv")
    per_path_rows = read_csv(bundle / "per_path.csv")
    statistics = group_statistics(config, specs, per_path_rows)
    statistics["routing_robustness"] = routing_classification(statistics)
    atomic_write_json(bundle / "statistics.json", statistics)
    step_count = combine_step_records(
        output_dir,
        bundle / "step_records.jsonl.gz",
        specs,
        config["source_policy_seeds"],
    )
    render_figures(statistics, bundle)
    readme = """# ABIDES Dose And Routing Sensitivity v1 Audit Bundle

This is the repo-visible audit export of the preregistered secondary
sensitivity panel. It contains three realized-dose groups and five unique
routing-intensity groups. The 10-policy-seed panel must not replace the v5
20-seed primary result or be used to select a new primary operating point.

Run `python revision/experiments/verify_abides_dose_routing_sensitivity.py`
to recompute per-seed aggregates, paired t intervals, exact sign tests,
two-way paired bootstrap intervals, Holm adjustment, budget diagnostics,
per-step path integrity, and the frozen routing classification.
"""
    atomic_write_text(bundle / "README.md", readme)

    source_by_seed: dict[str, Any] = {}
    task_artifacts: dict[str, Any] = {}
    for result in results:
        seed = str(result["policy_seed"])
        source_by_seed.setdefault(
            seed,
            {
                "source_checkpoint_sha256": result["source_checkpoint_sha256"],
                "base_model_digest": result["base_model_digest"],
                "initial_adapter_digest": result["initial_adapter_digest"],
            },
        )
        key = f"{result['group_spec']['group_id']}:{seed}"
        task_artifacts[key] = {
            "final_model_digests": result["final_model_digests"],
            "step_records_sha256": result["step_records_sha256"],
            "completed_episodes": result["completed_episodes"],
        }
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).splitlines()
    source_files = (
        "aggregate.json",
        "completion.json",
        "per_path.csv",
        "per_seed.csv",
        "statistics.json",
        "step_records.jsonl.gz",
        "abides_dose_curve.png",
        "abides_routing_sensitivity.png",
        "README.md",
    )
    manifest = {
        "protocol_version": output_manifest["protocol_version"],
        "protocol_fingerprint": output_manifest["protocol_fingerprint"],
        "protocol_freeze_commit": output_manifest["execution_commit"],
        "execution_commit": output_manifest["execution_commit"],
        "result_archival_commit": validate_archival_commit(result_archival_commit),
        "config": output_manifest["config"],
        "config_sha256": output_manifest["config_sha256"],
        "calibration_artifact_sha256": output_manifest[
            "calibration_artifact_sha256"
        ],
        "source_v5_fingerprint": config["source_base_fingerprint"],
        "source_v5_manifest_sha256": config["source_base_manifest_sha256"],
        "source_policy_artifacts": source_by_seed,
        "task_artifacts": task_artifacts,
        "abides_commit": ABIDES_COMMIT,
        "docker_image_id": output_manifest["docker_image_id"],
        "docker_oci_digest": None,
        "host_software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "seed_namespaces": {
            key: config[key]
            for key in (
                "source_policy_seeds",
                "passive_calibration_seeds",
                "evaluation_seeds",
                "response_environment_seed_namespace",
                "response_action_namespace",
                "evaluation_action_namespace",
                "bootstrap_seed",
            )
        },
        "started_at": output_manifest["started_at"],
        "finished_at": completion["finished_at"],
        "completed_tasks": completion["completed_tasks"],
        "required_tasks": completion["required_tasks"],
        "pending_tasks": completion["required_tasks"] - completion["completed_tasks"],
        "failed_tasks": completion["failed_tasks_this_invocation"],
        "step_record_count": step_count,
        "implementation_status": aggregate["implementation_status"],
        "scientific_result": aggregate["scientific_result"],
        "routing_robustness": statistics["routing_robustness"],
        "completed_resume_verified": bool(
            completion["completed_resume_verification"]
            and completion["aggregate_unchanged_on_completed_resume"]
            and int(completion["containers_started"]) == 0
        ),
        "export_git_dirty_paths": status,
        "output_file_sha256": {
            name: sha256_file(bundle / name) for name in source_files
        },
        "command": (
            "python revision/experiments/run_abides_dose_routing_sensitivity.py "
            "--config revision/configs/abides_dose_routing_sensitivity_v1.json"
        ),
    }
    atomic_write_json(bundle / "manifest.json", manifest)
    checksum_files = (*source_files, "manifest.json")
    atomic_write_text(
        bundle / "SHA256SUMS",
        "".join(
            f"{sha256_file(bundle / name)}  {name}\n" for name in checksum_files
        ),
    )
    for source_name, paper_path in (
        ("per_path.csv", REPO / PAPER_PER_PATH),
        ("per_seed.csv", REPO / PAPER_PER_SEED),
        ("abides_dose_curve.png", REPO / PAPER_DOSE_FIGURE),
        ("abides_routing_sensitivity.png", REPO / PAPER_ROUTING_FIGURE),
    ):
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / source_name, paper_path)
    return manifest


def check_export(bundle: Path) -> None:
    checksums = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in checksums:
        expected, name = line.split("  ", 1)
        if sha256_file(bundle / name) != expected:
            raise RuntimeError(f"audit checksum mismatch: {name}")
    for source_name, paper_path in (
        ("per_path.csv", REPO / PAPER_PER_PATH),
        ("per_seed.csv", REPO / PAPER_PER_SEED),
        ("abides_dose_curve.png", REPO / PAPER_DOSE_FIGURE),
        ("abides_routing_sensitivity.png", REPO / PAPER_ROUTING_FIGURE),
    ):
        if (bundle / source_name).read_bytes() != paper_path.read_bytes():
            raise RuntimeError(f"paper export differs from audit source: {paper_path}")
    print(canonical_json({"status": "export_check_pass", "bundle": str(bundle)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--result-archival-commit")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bundle = REPO / args.bundle
    if args.check:
        check_export(bundle)
        return
    manifest = export_bundle(
        REPO / args.config,
        REPO / args.output,
        bundle,
        args.result_archival_commit,
    )
    print(
        canonical_json(
            {
                "status": "exported",
                "bundle": str(bundle.relative_to(REPO)),
                "protocol_fingerprint": manifest["protocol_fingerprint"],
            }
        )
    )


if __name__ == "__main__":
    main()
