"""Command-line entry points for JSONL processing and synthetic evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from schema_evolution.benchmark import run_benchmark
from schema_evolution.pipeline import PipelineEngine, source_name
from schema_evolution.synthetic import generate_dataset

logger = logging.getLogger("schema_evolution")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            yield value


def collect_baseline(path: Path, records_per_source: int) -> dict[str, list[dict[str, Any]]]:
    if records_per_source < 2:
        raise ValueError("--baseline-records-per-source must be at least 2")
    baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals: Counter[str] = Counter()
    for record in iter_jsonl(path):
        source = source_name(record)
        totals[source] += 1
        if len(baseline[source]) < records_per_source:
            baseline[source].append(record)
    if not totals:
        raise ValueError("Input JSONL file contains no records")
    insufficient = sorted(source for source, count in totals.items() if count < records_per_source)
    if insufficient:
        raise ValueError("Not enough baseline records for source(s): " + ", ".join(insufficient))
    return dict(baseline)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")
    os.replace(temporary, path)


def process_file(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    baseline_records_per_source: int = 50,
    seed: int = 17,
    confirmations: int = 2,
) -> dict[str, Any]:
    if input_path.resolve() in {output_path.resolve(), report_path.resolve()}:
        raise ValueError("Input, cleaned output, and report paths must be different")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("Cleaned output and report paths must be different")
    baselines = collect_baseline(input_path, baseline_records_per_source)
    baseline_records = [record for records in baselines.values() for record in records]
    engine = PipelineEngine.fit(
        baseline_records, seed=seed, confirmations=confirmations, tune_parameters=True
    )
    remaining = {source: len(records) for source, records in baselines.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    latencies: list[float] = []
    correction_counts: Counter[str] = Counter()
    drift_events = accepted_versions = quarantined_records = processed_records = 0
    with temporary_output.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in iter_jsonl(input_path):
            source = source_name(record)
            if remaining.get(source, 0):
                remaining[source] -= 1
                continue
            started = time.perf_counter_ns()
            result = engine.process(record)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            processed_records += 1
            if result.drift:
                drift_events += 1
                accepted_versions += int(result.drift.accepted)
            quarantined_records += int(result.quarantined)
            for decision in result.corrections:
                correction_counts[f"{decision.status}:{decision.rule}"] += 1
            output_file.write(
                json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            )
    os.replace(temporary_output, output_path)
    if not latencies:
        report = {
            "input_records": sum(len(records) for records in baselines.values()),
            "processed_records": 0,
            "message": "Input contained baseline records only; no stream records were processed.",
        }
    else:
        ordered = sorted(latencies)

        def percentile(fraction: float) -> float:
            return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

        report = {
            "input_records": sum(len(records) for records in baselines.values())
            + processed_records,
            "baseline_records_per_source": baseline_records_per_source,
            "baseline_sources": sorted(baselines),
            "processed_records": processed_records,
            "schema_drift_events": drift_events,
            "accepted_schema_versions": accepted_versions,
            "quarantined_records": quarantined_records,
            "correction_decisions": dict(correction_counts),
            "latency_ms_p50": round(statistics.median(latencies), 4),
            "latency_ms_p95": round(percentile(0.95), 4),
            "latency_ms_p99": round(percentile(0.99), 4),
            "latency_note": "Local single-process measurement; not a capacity guarantee.",
        }
    _write_json_atomic(report_path, report)
    return report


def _write_synthetic(path: Path, baseline_count: int, test_count: int, seed: int) -> None:
    dataset = generate_dataset(
        records_per_source=baseline_count, test_per_source=test_count, seed=seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    truth_path = path.with_name(path.stem + ".ground-truth.jsonl")
    with (
        path.open("w", encoding="utf-8", newline="\n") as raw_file,
        truth_path.open("w", encoding="utf-8", newline="\n") as truth_file,
    ):
        for record in dataset.baseline:
            raw_file.write(json.dumps(record) + "\n")
        test_indices: Counter[str] = Counter()
        for case in dataset.cases:
            raw_file.write(json.dumps(case.raw) + "\n")
            source = str(case.raw["_source"])
            test_indices[source] += 1
            truth_file.write(
                json.dumps(
                    {
                        "source": source,
                        "test_index": test_indices[source],
                        "truth": case.truth,
                        "schema_changed": case.schema_changed,
                        "corrupted_fields": case.corrupted_fields,
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="schema-evolution")
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logging")
    commands = parser.add_subparsers(dest="command")

    process = commands.add_parser("process", help="Process a JSONL stream after per-source warm-up")
    process.add_argument("input", type=Path)
    process.add_argument("--output", type=Path, default=Path("cleaned.jsonl"))
    process.add_argument("--report", type=Path, default=Path("report.json"))
    process.add_argument("--baseline-records-per-source", type=int, default=50)
    process.add_argument("--confirmations", type=int, default=2)
    process.add_argument("--seed", type=int, default=17)

    benchmark = commands.add_parser("benchmark", help="Run the seeded synthetic benchmark")
    benchmark.add_argument("--trials", type=int, default=3)
    benchmark.add_argument("--records-per-source", type=int, default=100)
    benchmark.add_argument("--test-per-source", type=int, default=120)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--output", type=Path, default=Path("benchmark-report.json"))

    synthetic = commands.add_parser(
        "generate-synthetic", help="Write synthetic JSONL and ground truth"
    )
    synthetic.add_argument("output", type=Path)
    synthetic.add_argument("--baseline-count", type=int, default=100)
    synthetic.add_argument("--test-count", type=int, default=120)
    synthetic.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.command is None:
        from schema_evolution.interactive import run_interactive

        run_interactive()
        return

    try:
        if args.command == "process":
            report = process_file(
                args.input,
                args.output,
                args.report,
                baseline_records_per_source=args.baseline_records_per_source,
                seed=args.seed,
                confirmations=args.confirmations,
            )
        elif args.command == "benchmark":
            report = run_benchmark(
                trials=args.trials,
                records_per_source=args.records_per_source,
                test_per_source=args.test_per_source,
                seed=args.seed,
            )
            _write_json_atomic(args.output, report)
        else:
            _write_synthetic(args.output, args.baseline_count, args.test_count, args.seed)
            report = {"synthetic_jsonl": str(args.output), "seed": args.seed}
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
