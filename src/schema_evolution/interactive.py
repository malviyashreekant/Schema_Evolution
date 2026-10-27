"""Guided terminal interface for common schema-evolution workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _header() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]SCHEMA EVOLUTION[/bold cyan]\n"
            "[dim]Schema monitoring  ·  Quality repair  ·  Synthetic evaluation[/dim]",
            border_style="cyan",
        )
    )


def _ask_text(
    prompt: str,
    default: str,
    *,
    validator: Any = None,
) -> str | None:
    answer = questionary.text(prompt, default=default, validate=validator).ask()
    return answer.strip() if isinstance(answer, str) else None


def _ask_integer(prompt: str, default: int, minimum: int = 1) -> int | None:
    def validate(value: str) -> bool | str:
        if not value.isdigit() or int(value) < minimum:
            return f"Enter a whole number greater than or equal to {minimum}."
        return True

    answer = _ask_text(prompt, str(default), validator=validate)
    return int(answer) if answer is not None else None


def _input_file(value: str) -> bool | str:
    if not value.strip():
        return "Enter a JSONL file path."
    if not Path(value).expanduser().is_file():
        return "That file does not exist. Enter a valid path."
    return True


def _render_process_report(report: dict[str, Any]) -> None:
    table = Table(title="Processing summary", header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Result", justify="right")
    for label, key in (
        ("Input records", "input_records"),
        ("Processed records", "processed_records"),
        ("Schema drift events", "schema_drift_events"),
        ("Accepted schema versions", "accepted_schema_versions"),
        ("Quarantined records", "quarantined_records"),
    ):
        if key in report:
            table.add_row(label, str(report[key]))
    if "latency_ms_p95" in report:
        table.add_row("p95 processing latency", f"{report['latency_ms_p95']:.3f} ms")
    console.print(table)
    if report.get("correction_decisions"):
        corrections = Table(title="Correction decisions", header_style="bold cyan")
        corrections.add_column("Decision")
        corrections.add_column("Count", justify="right")
        for label, count in sorted(report["correction_decisions"].items()):
            corrections.add_row(label, str(count))
        console.print(corrections)


def _render_benchmark_report(report: dict[str, Any]) -> None:
    table = Table(title="Synthetic benchmark", header_style="bold cyan")
    table.add_column("Method")
    table.add_column("Repair precision", justify="right")
    table.add_column("Repair recall", justify="right")
    table.add_column("Completeness", justify="right")
    table.add_column("Schema adaptability", justify="right")
    table.add_column("p95 latency", justify="right")
    for method, metrics in report["methods"].items():
        table.add_row(
            method.replace("_", " "),
            f"{metrics['repair_precision']:.1%}",
            f"{metrics['repair_recall']:.1%}",
            f"{metrics['completeness']:.1%}",
            f"{metrics['schema_adaptability']:.1%}",
            f"{metrics['latency_ms_p95']:.3f} ms",
        )
    console.print(table)
    console.print(
        "[dim]Synthetic results are for comparison only; they are not production "
        "performance claims.[/dim]"
    )


def _process_workflow() -> None:
    from schema_evolution.cli import process_file

    input_path = _ask_text("Input JSONL file", "sample_stream.jsonl", validator=_input_file)
    if input_path is None:
        return
    suggested_baseline = 3 if Path(input_path).name == "sample_stream.jsonl" else 50
    baseline_count = _ask_integer("Baseline records per source", suggested_baseline, minimum=2)
    if baseline_count is None:
        return
    confirmations = _ask_integer("Confirmations before accepting schema drift", 2)
    if confirmations is None:
        return
    output_path = _ask_text("Cleaned audit JSONL output", "cleaned.jsonl")
    if output_path is None:
        return
    report_path = _ask_text("Summary report JSON output", "report.json")
    if report_path is None:
        return
    report = process_file(
        Path(input_path).expanduser(),
        Path(output_path).expanduser(),
        Path(report_path).expanduser(),
        baseline_records_per_source=baseline_count,
        confirmations=confirmations,
    )
    _render_process_report(report)
    console.print(f"Cleaned audit records: [cyan]{output_path}[/cyan]")
    console.print(f"Summary report: [cyan]{report_path}[/cyan]")


def _synthetic_workflow() -> None:
    from schema_evolution.cli import _write_synthetic

    output_path = _ask_text("Synthetic JSONL output", "generated.jsonl")
    if output_path is None:
        return
    baseline_count = _ask_integer("Baseline records per source", 100, minimum=10)
    if baseline_count is None:
        return
    test_count = _ask_integer("Test records per source", 120, minimum=10)
    if test_count is None:
        return
    seed = _ask_integer("Random seed", 42, minimum=0)
    if seed is None:
        return
    path = Path(output_path).expanduser()
    _write_synthetic(path, baseline_count, test_count, seed)
    truth_path = path.with_name(path.stem + ".ground-truth.jsonl")
    console.print(f"Synthetic stream: [cyan]{path}[/cyan]")
    console.print(f"Ground truth: [cyan]{truth_path}[/cyan]")


def _benchmark_workflow() -> None:
    from schema_evolution.benchmark import run_benchmark
    from schema_evolution.cli import _write_json_atomic

    trials = _ask_integer("Number of trials", 3)
    if trials is None:
        return
    baseline_count = _ask_integer("Baseline records per source per trial", 100, minimum=10)
    if baseline_count is None:
        return
    test_count = _ask_integer("Test records per source per trial", 120, minimum=10)
    if test_count is None:
        return
    seed = _ask_integer("Starting random seed", 42, minimum=0)
    if seed is None:
        return
    output_path = _ask_text("Benchmark report output", "artifacts/benchmark-report.json")
    if output_path is None:
        return
    with console.status("Running synthetic trials..."):
        report = run_benchmark(
            trials=trials,
            records_per_source=baseline_count,
            test_per_source=test_count,
            seed=seed,
        )
    path = Path(output_path).expanduser()
    _write_json_atomic(path, report)
    _render_benchmark_report(report)
    console.print(f"Full benchmark report: [cyan]{path}[/cyan]")


def run_interactive() -> None:
    """Show a guided menu and run the selected workflow."""
    _header()
    choices = [
        questionary.Choice("Process a JSONL data file", value="process"),
        questionary.Choice("Generate synthetic test data", value="synthetic"),
        questionary.Choice("Run the synthetic benchmark", value="benchmark"),
        questionary.Choice("Exit", value="exit"),
    ]
    actions = {
        "process": _process_workflow,
        "synthetic": _synthetic_workflow,
        "benchmark": _benchmark_workflow,
    }
    try:
        while True:
            selection = questionary.select("What would you like to do?", choices=choices).ask()
            if selection is None or selection == "exit":
                console.print("Goodbye.")
                return
            try:
                actions[selection]()
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                console.print(f"[bold red]Could not complete that task:[/bold red] {error}")
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
