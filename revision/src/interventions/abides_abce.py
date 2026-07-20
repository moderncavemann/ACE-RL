"""Pure schedule and identity controls for the ABCE follow-up."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def signal_multiset_digest(values: Sequence[float]) -> str:
    return canonical_digest(sorted(float(value) for value in values))


@dataclass(frozen=True)
class ScheduleTransform:
    values: tuple[float, ...]
    source_steps: tuple[int, ...]
    transform_id: str

    def exact_multiset_of(self, source: Sequence[float]) -> bool:
        return bool(
            np.allclose(
                np.sort(np.asarray(self.values, dtype=float)),
                np.sort(np.asarray(source, dtype=float)),
                atol=1e-12,
                rtol=0.0,
            )
        )


def aligned_schedule(values: Sequence[float]) -> ScheduleTransform:
    normalized = tuple(float(value) for value in values)
    return ScheduleTransform(
        values=normalized,
        source_steps=tuple(range(len(normalized))),
        transform_id="aligned",
    )


def exact_random_schedule(values: Sequence[float], seed: int) -> ScheduleTransform:
    normalized = np.asarray(values, dtype=float)
    source_steps = np.random.default_rng(int(seed)).permutation(len(normalized))
    shuffled = normalized[source_steps]
    return ScheduleTransform(
        values=tuple(float(value) for value in shuffled),
        source_steps=tuple(int(value) for value in source_steps),
        transform_id=f"exact_random:{int(seed)}",
    )


def circular_schedule(values: Sequence[float], seed: int) -> ScheduleTransform:
    normalized = np.asarray(values, dtype=float)
    if len(normalized) < 2:
        raise ValueError("circular schedule requires at least two decisions")
    offset = int(np.random.default_rng(int(seed)).integers(1, len(normalized)))
    source_steps = np.roll(np.arange(len(normalized), dtype=int), offset)
    shifted = normalized[source_steps]
    return ScheduleTransform(
        values=tuple(float(value) for value in shifted),
        source_steps=tuple(int(value) for value in source_steps),
        transform_id=f"circular_shift:{offset}",
    )


def periodic_schedule(values: Sequence[float]) -> ScheduleTransform:
    normalized = np.asarray(values, dtype=float)
    active_source_steps = np.flatnonzero(normalized > 0.0)
    result = np.zeros_like(normalized)
    source_steps = np.full(len(normalized), -1, dtype=int)
    if len(active_source_steps):
        positions = np.floor(
            (np.arange(len(active_source_steps), dtype=float) + 0.5)
            * len(normalized)
            / len(active_source_steps)
        ).astype(int)
        if len(np.unique(positions)) != len(positions):
            raise RuntimeError("periodic placement produced duplicate decisions")
        result[positions] = normalized[active_source_steps]
        source_steps[positions] = active_source_steps
    return ScheduleTransform(
        values=tuple(float(value) for value in result),
        source_steps=tuple(int(value) for value in source_steps),
        transform_id="periodic_even_placement",
    )


def lagged_dealer_schedule(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        return ()
    return (0.0, *normalized[:-1])


def frozen_identity_permutation(
    n_agents: int,
    environment_seed: int,
    step: int,
    namespace: int,
) -> tuple[int, ...]:
    if n_agents < 2:
        raise ValueError("identity permutation requires at least two dealers")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(environment_seed), int(step), int(namespace)]
        )
    )
    permutation = rng.permutation(n_agents)
    if np.array_equal(permutation, np.arange(n_agents)):
        permutation = np.roll(permutation, 1)
    return tuple(int(value) for value in permutation)


def identity_permutation_schedule(
    n_agents: int,
    environment_seed: int,
    episode_steps: int,
    namespace: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        frozen_identity_permutation(
            n_agents,
            environment_seed,
            step,
            namespace,
        )
        for step in range(episode_steps)
    )
