"""Repeatable synthetic benchmark with no-op and static-rule baselines."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from typing import Any

from schema_evolution.fuzzy_quality import build_profiles
from schema_evolution.pipeline import PipelineEngine
from schema_evolution.synthetic import SyntheticCase, generate_dataset


def _percentile(values: list[float], quantile: float) -> float:
    return (
        float(statistics.quantiles(values, n=100, method="inclusive")[round(quantile * 100) - 1])
        if len(values) > 1
        else values[0]
    )


def _evaluate_method(
    baseline: list[dict[str, Any]], cases: list[SyntheticCase], method: str, seed: int
) -> dict[str, Any]:
    if method == "no_op":
        engine = None
    elif method == "static_rules":
        engine = PipelineEngine.fit(
            baseline,
            seed=seed,
            tune_parameters=False,
            fuzzy_abstention=False,
            detect_schema=False,
        )
    else:
        engine = PipelineEngine.fit(baseline, seed=seed, tune_parameters=True)

    latencies: list[float] = []
    corrupt_total = corrected_total = correct_edits = 0
    missing_total = missing_filled = 0
    expected_sources = {
        str(case.raw.get("_source", "default")) for case in cases if case.schema_changed
    }
    detected_sources: set[str] = set()
    accepted_sources: set[str] = set()
    records_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in baseline:
        records_by_source[str(record.get("_source", "default"))].append(record)
    profiles = {
        source: build_profiles(records) for source, records in records_by_source.items()
    }
    for case in cases:
        started = time.perf_counter_ns()
        result = engine.process(case.raw) if engine else None
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        cleaned = result.cleaned if result else case.raw
        source = str(case.raw.get("_source", "default"))
        if case.schema_changed and result and result.drift:
            detected_sources.add(source)
            if result.drift.accepted:
                accepted_sources.add(source)
        for field in case.corrupted_fields:
            corrupt_total += 1
            if cleaned.get(field) != case.raw.get(field):
                corrected_total += 1
                expected = case.truth.get(field)
                actual = cleaned.get(field)
                profile = profiles.get(source, {}).get(field)
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    tolerance = max(0.25, profile.robust_scale * 1.5) if profile else 0.25
                    matches = (
                        isinstance(actual, (int, float))
                        and abs(actual - expected) <= tolerance
                    )
                else:
                    matches = actual == expected
                if matches:
                    correct_edits += 1
            if field in case.missing_fields:
                missing_total += 1
                if cleaned.get(field) is not None:
                    missing_filled += 1
    return {
        "records": len(cases),
        "corrupted_fields": corrupt_total,
        "corrected_fields": corrected_total,
        "correctly_repaired_fields": correct_edits,
        "repair_recall": correct_edits / max(1, corrupt_total),
        "repair_precision": correct_edits / max(1, corrected_total),
        "missing_fields": missing_total,
        "completeness": missing_filled / max(1, missing_total),
        "schema_transitions_expected": len(expected_sources),
        "schema_transitions_detected": len(detected_sources & expected_sources),
        "schema_drift_detection_rate": len(detected_sources & expected_sources)
        / max(1, len(expected_sources)),
        "schema_versions_accepted": len(accepted_sources),
        "schema_adaptability": len(accepted_sources & expected_sources)
        / max(1, len(expected_sources)),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_p99": _percentile(latencies, 0.99),
    }


def run_benchmark(
    *, trials: int = 3, records_per_source: int = 100, test_per_source: int = 120, seed: int = 42
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    methods = ("no_op", "static_rules", "proposed_mlp_fuzzy_ga")
    trial_results: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for trial in range(trials):
        trial_seed = seed + trial
        dataset = generate_dataset(
            records_per_source=records_per_source,
            test_per_source=test_per_source,
            seed=trial_seed,
        )
        for method in methods:
            trial_results[method].append(
                _evaluate_method(dataset.baseline, dataset.cases, method, trial_seed)
            )
    summary: dict[str, Any] = {}
    for method, runs in trial_results.items():
        metric_names = [
            "repair_recall",
            "repair_precision",
            "completeness",
            "schema_drift_detection_rate",
            "schema_adaptability",
            "schema_versions_accepted",
            "latency_ms_p50",
            "latency_ms_p95",
            "latency_ms_p99",
        ]
        summary[method] = {
            metric: round(statistics.mean(run[metric] for run in runs), 6)
            for metric in metric_names
        }
    return {
        "benchmark": "synthetic_schema_evolution_v1",
        "seed": seed,
        "trials": trials,
        "records_per_source": records_per_source,
        "test_records_per_source": test_per_source,
        "training_test_separation": "Baseline records train models; test records are held out.",
        "methods": summary,
        "trial_results": trial_results,
        "limitations": [
            "Synthetic distributions are not representative of any production domain.",
            "The GA uses generated calibration corruptions from baseline values.",
            "Latency is a local single-process measurement, not a capacity guarantee.",
        ],
    }
