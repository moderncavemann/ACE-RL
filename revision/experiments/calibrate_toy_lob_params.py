#!/usr/bin/env python3
"""Calibrate canonical toy LOB parameters before observation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs.toy_lob_pricing_env import ToyLOBConfig
from src.utils.benchmarks import benchmark_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="revision/experiments/outputs/calibration.json")
    args = parser.parse_args()

    config = ToyLOBConfig()
    table = benchmark_table(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print(json.dumps(table, indent=2))
    print(f"\nSaved calibration to {output}")


if __name__ == "__main__":
    main()
