#!/usr/bin/env python3
"""Export the self-contained ABCE primary follow-up audit bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_abides_abce_primary_followup import (  # noqa: E402
    CONTRASTS,
    HOLM_FAMILY,
    METHODS,
    canonical_json,
    result_path,
    write_csv,
)
from run_abides_mappo_full_main import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from src.utils.abides_abce_audit import (  # noqa: E402
    aggregate_per_seed,
    build_statistics,
)


def load_results(config: Mapping[str, Any], runtime: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in config["policy_seeds"]:
        path = result_path(runtime, int(seed))
        if not path.exists():
            raise FileNotFoundError(f"missing ABCE result: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def copy_protocol_files(config: Mapping[str, Any], artifact: Path) -> None:
    shutil.copyfile(REPO / str(config["protocol_file"]), artifact / "PROTOCOL.md")
    shutil.copyfile(
        REPO / str(config["execution_amendment_file"]),
        artifact / "EXECUTION_AMENDMENT.md",
    )


def concatenate_steps(
    results: Sequence[Mapping[str, Any]], runtime: Path, destination: Path
) -> tuple[int, dict[str, bool]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    checks = {
        "trace_no_lookahead": True,
        "trace_weights_sum_to_one": True,
        "trace_fixed_volume": True,
        "trace_identity_weight_multiset_exact": True,
        "trace_finite": True,
    }
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as target:
                for result in sorted(results, key=lambda row: int(row["policy_seed"])):
                    source = REPO / str(result["step_records_file"])
                    if not source.is_relative_to(runtime):
                        raise RuntimeError("ABCE step source escaped runtime output")
                    if sha256_file(source) != str(result["step_records_sha256"]):
                        raise RuntimeError(f"ABCE raw step hash mismatch: {source}")
                    with gzip.open(source, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            row = json.loads(line)
                            integrity = row["integrity"]
                            checks["trace_no_lookahead"] &= bool(
                                integrity["no_lookahead"]
                            )
                            checks["trace_weights_sum_to_one"] &= bool(
                                integrity["weights_sum_to_one"]
                            )
                            checks["trace_fixed_volume"] &= bool(
                                integrity["fixed_volume"]
                            )
                            checks["trace_identity_weight_multiset_exact"] &= bool(
                                integrity["identity_weight_multiset_exact"]
                            )
                            checks["trace_finite"] &= bool(integrity["finite"])
                            target.write(canonical_json(row) + "\n")
                            count += 1
    temporary.replace(destination)
    return count, checks


def representative_selection(
    per_path: Sequence[Mapping[str, Any]], per_seed: Sequence[Mapping[str, Any]]
) -> tuple[int, int]:
    effects = {
        int(row["policy_seed"]): float(
            row["exact_random_soft_rank__low_execution_half_spread"]
        )
        - float(row["aligned_soft_rank__low_execution_half_spread"])
        for row in per_seed
    }
    central = median(effects.values())
    policy_seed = min(effects, key=lambda seed: (abs(effects[seed] - central), seed))
    lookup = {
        (int(row["policy_seed"]), int(row["path_seed"]), str(row["method"])): row
        for row in per_path
    }
    path_seeds = sorted(
        {
            int(row["path_seed"])
            for row in per_path
            if int(row["policy_seed"]) == policy_seed
        }
    )
    path_effects = {
        path: float(
            lookup[(policy_seed, path, "exact_random_soft_rank")][
                "low_execution_half_spread"
            ]
        )
        - float(
            lookup[(policy_seed, path, "aligned_soft_rank")][
                "low_execution_half_spread"
            ]
        )
        for path in path_seeds
    }
    path_central = median(path_effects.values())
    path_seed = min(
        path_effects,
        key=lambda path: (abs(path_effects[path] - path_central), path),
    )
    return policy_seed, path_seed


def load_selected_traces(
    destination: Path, policy_seed: int, path_seed: int
) -> dict[str, list[dict[str, Any]]]:
    selected_methods = (
        "aligned_soft_rank",
        "exact_random_soft_rank",
        "lagged_revelation_soft_rank",
        "identity_permuted_soft_rank",
        "no_routing",
    )
    result = {method: [] for method in selected_methods}
    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            provenance = row["provenance"]
            if (
                int(provenance["policy_seed"]) == policy_seed
                and int(provenance["path_seed"]) == path_seed
                and str(provenance["method"]) in selected_methods
            ):
                result[str(provenance["method"])].append(row)
    for method in result:
        result[method].sort(key=lambda row: int(row["provenance"]["step"]))
    return result


def plot_representative_trace(
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    policy_seed: int,
    path_seed: int,
    figure_dir: Path,
) -> None:
    labels = {
        "aligned_soft_rank": "Aligned",
        "exact_random_soft_rank": "Exact random",
        "lagged_revelation_soft_rank": "Lagged reveal",
        "identity_permuted_soft_rank": "Identity permuted",
        "no_routing": "No routing",
    }
    colors = {
        "aligned_soft_rank": "#146B5D",
        "exact_random_soft_rank": "#C44E52",
        "lagged_revelation_soft_rank": "#4C72B0",
        "identity_permuted_soft_rank": "#8172B2",
        "no_routing": "#5A5A5A",
    }
    aligned = traces["aligned_soft_rank"]
    steps = np.arange(len(aligned))
    risk = np.asarray(
        [row["estimator"]["latent_risk"]["value"] == "high" for row in aligned],
        dtype=float,
    )
    score = np.asarray([row["estimator"]["proxy_score"] for row in aligned], dtype=float)
    fig, axes = plt.subplots(5, 1, figsize=(7.1, 7.8), sharex=True)
    axes[0].fill_between(steps, 0.0, risk, color="#D9D9D9", step="post", label="High risk")
    axes[0].plot(steps, score, color="#111111", linewidth=1.2, label="Public score")
    axes[0].axhline(float(aligned[0]["estimator"]["threshold"]), color="#A33", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Risk / score")
    for method, rows in traces.items():
        applied = [row["scheduler"]["platform_applied_signal"] for row in rows]
        axes[1].plot(steps, applied, color=colors[method], linewidth=1.0, label=labels[method])
        if method == "lagged_revelation_soft_rank":
            observed = [row["scheduler"]["dealer_observed_signal"] for row in rows]
            axes[1].plot(steps, observed, color=colors[method], linestyle=":", linewidth=1.0)
        axes[2].plot(
            steps,
            [row["allocation"]["routing_total_variation"] for row in rows],
            color=colors[method],
            linewidth=1.0,
        )
        half_spreads = np.asarray([row["dealer"]["half_spreads"] for row in rows], dtype=float)
        axes[3].plot(steps, np.mean(half_spreads, axis=1), color=colors[method], linewidth=1.0)
        axes[3].plot(
            steps,
            np.min(half_spreads, axis=1),
            color=colors[method],
            linewidth=0.7,
            linestyle=":",
            alpha=0.75,
        )
        axes[4].plot(
            steps,
            [
                row["outcomes"]["customer_execution_cost_tick_units"]
                / max(sum(item["quantity"] for item in row["outcomes"]["executions"]), 1)
                for row in rows
            ],
            color=colors[method],
            linewidth=1.0,
        )
    axes[1].set_ylabel("Contract")
    axes[2].set_ylabel("Routing TV")
    axes[3].set_ylabel("Mean / best quote")
    axes[4].set_ylabel("Exec. spread")
    axes[4].set_xlabel("Decision")
    axes[1].legend(ncol=3, frameon=False, fontsize=7, loc="upper right")
    for axis in axes:
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Preregistered median trace: policy {policy_seed}, path {path_seed}", fontsize=10)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figure_dir / "representative_trace.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "ACE-RL ABCE v1",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        figure_dir / "representative_trace.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "ACE-RL ABCE v1"},
    )
    plt.close(fig)


def latex_table(statistics: Mapping[str, Any], destination: Path) -> None:
    rows = []
    for name, result in statistics["contrasts"].items():
        interval = result["t_interval"]
        label = name.replace("_", "\\_")
        rows.append(
            f"{label} & {float(interval['mean']):.3f} & "
            f"[{float(interval['ci_low']):.3f}, {float(interval['ci_high']):.3f}] & "
            f"{int(interval['positive'])}/{int(interval['n'])} \\\\"
        )
    text = "\n".join(
        [
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Contrast & Mean $\\Delta$ & 95\\% CI & Positive seeds \\\\ ",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    atomic_write_text(destination, text)


def latex_method_table(statistics: Mapping[str, Any], destination: Path) -> None:
    labels = {
        "aligned_soft_rank": "Aligned soft-rank",
        "exact_random_soft_rank": "Exact random",
        "circular_shifted_soft_rank": "Circular shift",
        "periodic_exact_multiset_soft_rank": "Periodic",
        "lagged_revelation_soft_rank": "Lagged reveal",
        "identity_permuted_soft_rank": "Identity permuted",
        "hard_best_aligned": "Hard-best aligned",
        "hard_best_exact_random": "Hard-best random",
        "no_routing": "No routing",
    }
    rows = []
    for method in METHODS:
        summary = statistics["method_summary"][method]
        spread = summary["low_execution_half_spread"]
        reward = summary["aggregate_dealer_reward"]
        signal = summary["announced_signal_activation_rate"]
        routing = summary["mean_routing_total_variation"]
        rows.append(
            f"{labels[method]} & {spread['mean']:.3f} $\\pm$ "
            f"{spread['sample_sd']:.3f} & {reward['mean']:.2f} $\\pm$ "
            f"{reward['sample_sd']:.2f} & {signal['mean']:.3f} & "
            f"{routing['mean']:.3f} \\\\"
        )
    text = "\n".join(
        [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Method & $Y_L$ & Dealer reward & $A^{sig}$ & $\\overline{TV}$ \\\\ ",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    atomic_write_text(destination, text)


def sha256_manifest(artifact: Path) -> None:
    files = sorted(
        path
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(artifact)}" for path in files
    ]
    atomic_write_text(artifact / "SHA256SUMS", "\n".join(lines) + "\n")


def export(config_path: Path, artifact: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = REPO / str(config["output_dir"])
    summary = json.loads((runtime / "summary.json").read_text(encoding="utf-8"))
    runtime_manifest = json.loads(
        (runtime / "manifest.json").read_text(encoding="utf-8")
    )
    if summary["implementation_status"] != "pass":
        raise RuntimeError("cannot export an ABCE runtime that failed integrity")
    if artifact.exists():
        shutil.rmtree(artifact)
    artifact.mkdir(parents=True)
    copy_protocol_files(config, artifact)
    atomic_write_json(artifact / "protocol.json", config)
    results = load_results(config, runtime)
    per_path = [
        row
        for result in results
        for row in result["final_evaluation"]["per_path"]
    ]
    per_seed = aggregate_per_seed(per_path, METHODS)
    statistics = build_statistics(
        per_path,
        METHODS,
        CONTRASTS,
        HOLM_FAMILY,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    write_csv(artifact / "per_path.csv", per_path)
    write_csv(artifact / "per_seed.csv", per_seed)
    atomic_write_json(artifact / "statistics.json", statistics)
    trace_count, trace_checks = concatenate_steps(
        results, runtime, artifact / "step_records.jsonl.gz"
    )
    from verify_abides_abce_primary_followup_v1 import verify_trace

    independently_counted, verifier_trace_checks = verify_trace(artifact, config)
    if independently_counted != trace_count:
        raise RuntimeError("ABCE exported trace recount mismatch")
    expected_trace_count = (
        len(config["policy_seeds"])
        * len(config["evaluation_seeds"])
        * len(METHODS)
        * int(config["episode_steps"])
    )
    checks = {
        **dict(summary["checks"]),
        **trace_checks,
        **verifier_trace_checks,
        "trace_row_count": trace_count == expected_trace_count,
    }
    atomic_write_json(artifact / "checks.json", checks)
    atomic_write_json(
        artifact / "model_digests.json",
        {
            str(result["policy_seed"]): {
                "base": result["base_model_digest"],
                "methods": result["final_model_digests"],
            }
            for result in results
        },
    )
    elapsed = {
        str(result["policy_seed"]): float(result["elapsed_seconds"])
        for result in results
    }
    atomic_write_json(
        artifact / "efficiency.json",
        {
            "policy_seed_elapsed_seconds": elapsed,
            "mean_policy_seed_elapsed_seconds": float(np.mean(list(elapsed.values()))),
            "sample_sd_policy_seed_elapsed_seconds": (
                float(np.std(list(elapsed.values()), ddof=1))
                if len(elapsed) > 1
                else 0.0
            ),
            "total_policy_seed_compute_seconds": float(sum(elapsed.values())),
            "response_episodes_per_policy_seed": int(config["response_episodes"]),
            "methods_per_policy_seed": len(METHODS),
            "final_paths_per_method": len(config["evaluation_seeds"]),
            "configured_workers": int(config["workers"]),
            "device": str(config["device"]),
        },
    )
    atomic_write_json(
        artifact / "path_digests.json",
        {
            f"{row['policy_seed']}:{row['path_seed']}:{row['method']}": row[
                "path_digest"
            ]
            for row in per_path
        },
    )
    policy_seed, path_seed = representative_selection(per_path, per_seed)
    traces = load_selected_traces(
        artifact / "step_records.jsonl.gz", policy_seed, path_seed
    )
    plot_representative_trace(
        traces, policy_seed, path_seed, artifact / "figures"
    )
    (artifact / "tables").mkdir()
    latex_table(statistics, artifact / "tables" / "abce_contrasts.tex")
    latex_method_table(
        statistics, artifact / "tables" / "abce_method_summary.tex"
    )
    manifest = {
        "artifact_version": str(config["protocol_version"]),
        "created_at": runtime_manifest["started_at"],
        "runtime_manifest_sha256": sha256_file(runtime / "manifest.json"),
        "runtime_summary_sha256": sha256_file(runtime / "summary.json"),
        "runtime_result_sha256": {
            str(result["policy_seed"]): sha256_file(
                result_path(runtime, int(result["policy_seed"]))
            )
            for result in results
        },
        "trace_rows": trace_count,
        "representative_selection": {
            "rule": "closest to median A seed effect, then closest median path effect; ties lower seed",
            "policy_seed": policy_seed,
            "path_seed": path_seed,
        },
    }
    atomic_write_json(artifact / "manifest.json", manifest)
    atomic_write_text(
        artifact / "README.md",
        "# ABCE Primary Follow-up v1\n\n"
        "This bundle contains the frozen protocol, paired path and policy-seed "
        "tables, deterministic statistics, primitive step traces, integrity "
        "checks, model/path digests, and preregistered representative trace.\n",
    )
    sha256_manifest(artifact)
    return {"artifact": str(artifact), "checks": checks, **manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="revision/configs/abides_abce_primary_followup_v1.json",
    )
    parser.add_argument(
        "--artifact",
        default="revision/paper/artifacts/abides_abce_primary_followup_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export((REPO / args.config).resolve(), (REPO / args.artifact).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
