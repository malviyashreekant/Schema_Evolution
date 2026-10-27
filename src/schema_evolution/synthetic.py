"""Seeded synthetic multi-source records with clean ground truth and injected faults."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SyntheticCase:
    raw: dict[str, Any]
    truth: dict[str, Any]
    schema_changed: bool
    corrupted_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticDataset:
    baseline: list[dict[str, Any]]
    cases: list[SyntheticCase]
    seed: int


def _clean_record(source: str, rng: random.Random) -> dict[str, Any]:
    if source == "weather":
        return {
            "_source": source,
            "temperature": round(rng.gauss(20.0, 2.0), 2),
            "humidity": round(min(100.0, max(0.0, rng.gauss(52.0, 8.0))), 2),
            "station": rng.choice(["north", "south", "west"]),
        }
    return {
        "_source": source,
        "pressure": round(rng.gauss(10.0, 1.1), 2),
        "vibration": round(max(0.0, rng.gauss(0.55, 0.1)), 3),
        "status": rng.choice(["normal", "normal", "watch"]),
    }


def generate_dataset(
    *, records_per_source: int = 100, test_per_source: int = 120, seed: int = 42
) -> SyntheticDataset:
    """Generate clean training baselines and a separate corrupted test stream."""
    if records_per_source < 10 or test_per_source < 10:
        raise ValueError(
            "Synthetic generation requires at least 10 baseline and test records per source"
        )
    rng = random.Random(seed)
    sources = ("weather", "industrial")
    baseline = [_clean_record(source, rng) for source in sources for _ in range(records_per_source)]
    cases: list[SyntheticCase] = []
    numeric_fields = {"weather": "temperature", "industrial": "pressure"}
    for source in sources:
        changed_from = test_per_source // 2
        for index in range(test_per_source):
            truth = _clean_record(source, rng)
            schema_changed = index >= changed_from
            if schema_changed:
                evolved_field = "dew_point" if source == "weather" else "machine_mode"
                if source == "weather":
                    truth[evolved_field] = round(truth["temperature"] - 5.5, 2)
                else:
                    truth[evolved_field] = "standard"
            raw = dict(truth)
            field = numeric_fields[source]
            corrupted_fields: tuple[str, ...] = ()
            missing_fields: tuple[str, ...] = ()
            if index % 4 == 0:
                raw[field] = None
                corrupted_fields = (field,)
                missing_fields = (field,)
            elif index % 4 == 1:
                raw[field] = f"{truth[field]:.2f}"
                corrupted_fields = (field,)
            elif index % 4 == 2:
                raw[field] = round(float(truth[field]) + 60.0, 2)
                corrupted_fields = (field,)
            elif index % 4 == 3:
                category = "station" if source == "weather" else "status"
                raw[category] = "unrecognized-value"
                corrupted_fields = (category,)
            cases.append(
                SyntheticCase(raw, truth, schema_changed, corrupted_fields, missing_fields)
            )
    return SyntheticDataset(baseline, cases, seed)
