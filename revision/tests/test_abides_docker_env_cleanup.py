from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from src.envs.abides_docker_env import (  # noqa: E402
    ABIDESDockerConfig,
    ABIDESDockerDealerEnv,
)


class FakeDockerProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None and not self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("docker run", timeout)
        self.returncode = -9 if self.killed else -15
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_timeout_skips_second_request_and_force_removes_named_container() -> None:
    process = FakeDockerProcess()
    with (
        patch("src.envs.abides_docker_env.subprocess.Popen", return_value=process) as popen,
        patch("src.envs.abides_docker_env.select.select", return_value=([], [], [])),
        patch("src.envs.abides_docker_env.subprocess.run") as run,
    ):
        env = ABIDESDockerDealerEnv(
            {}, ABIDESDockerConfig(request_timeout_seconds=0.01)
        )
        with pytest.raises(TimeoutError):
            env.reset(seed=1)
        env.close()

    command = popen.call_args.args[0]
    name = command[command.index("--name") + 1]
    assert name.startswith("ace-rl-abides-")
    assert process.terminated
    run.assert_called_once()
    assert run.call_args.args[0] == ["docker", "rm", "-f", name]
