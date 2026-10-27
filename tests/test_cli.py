import json

from schema_evolution.cli import _write_synthetic, process_file


def test_generated_stream_runs_through_processing_cli(tmp_path):
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "cleaned.jsonl"
    report_path = tmp_path / "report.json"

    _write_synthetic(input_path, baseline_count=10, test_count=10, seed=12)
    report = process_file(
        input_path,
        output_path,
        report_path,
        baseline_records_per_source=10,
        seed=12,
    )

    cleaned_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    truth_path = input_path.with_name("synthetic.ground-truth.jsonl")
    truth_rows = [json.loads(line) for line in truth_path.read_text().splitlines()]
    assert report["processed_records"] == 20
    assert len(cleaned_rows) == len(truth_rows) == 20
    assert all("raw" in row and "cleaned" in row for row in cleaned_rows)
