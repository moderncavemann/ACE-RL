#!/usr/bin/env python3
"""Line-delimited JSON worker for a persistent ABIDES Docker process."""

import json
import sys
import traceback
from typing import Any, Dict

from ace_abides.economic_env import ABIDESDealerMarketConfig, ABIDESDealerMarketEnv


def write(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    sys.stdout.flush()


def config_from_dict(values: Dict[str, Any]) -> ABIDESDealerMarketConfig:
    data = dict(values)
    if "half_spread_levels" in data:
        data["half_spread_levels"] = tuple(int(x) for x in data["half_spread_levels"])
    return ABIDESDealerMarketConfig(**data)


def main() -> None:
    env = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                if env is not None:
                    env.close()
                config = config_from_dict(request.get("config", {}))
                env = ABIDESDealerMarketEnv(config)
                state = env.reset(seed=int(request["seed"]))
                write({"ok": True, "state": state})
            elif command == "step":
                if env is None:
                    raise RuntimeError("reset is required before step")
                state = env.step(
                    actions=[int(value) for value in request["actions"]],
                    gate_active=bool(request.get("gate_active", False)),
                    routing_enabled=bool(request.get("routing_enabled", True)),
                    routing_strength=(
                        None
                        if request.get("routing_strength") is None
                        else float(request["routing_strength"])
                    ),
                )
                write({"ok": True, "state": state})
            elif command == "close_episode":
                if env is not None:
                    env.close()
                    env = None
                write({"ok": True})
            elif command == "shutdown":
                if env is not None:
                    env.close()
                write({"ok": True})
                return
            else:
                raise ValueError(f"unsupported worker command: {command}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            write({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return


if __name__ == "__main__":
    main()
