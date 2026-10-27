# Schema Evolution Pipeline

An installable Python package and command-line tool for multi-source JSONL schema monitoring, confidence-scored quality correction, and reproducible synthetic evaluation.

## What it implements

- **Schema evolution:** per-source MLP classifier with two hidden layers, trained on baseline schemas and seeded synthetic mutations. Exact field/type differences remain visible in every event. A candidate schema is accepted after repeated confirmation; isolated changes are not silently promoted.
- **Data quality:** source-specific numeric/category profiles, Mamdani fuzzy inference, numeric string coercion, median/mode imputation, and robust outlier handling. Every proposed change carries its original value, replacement, rule, confidence, and status. Low-confidence changes abstain, preserve the raw value, and mark the record for review.
- **Genetic optimization:** DEAP evolves outlier sensitivity and minimum confidence using synthetic labeled corruptions derived only from baseline values. Tuning is performed during model setup, not for each incoming record.
- **Audit output:** each processed record includes both `raw` and `cleaned` values, schema evidence, correction decisions, and a quarantine flag.
- **Synthetic benchmark:** held-out records include missing values, numeric strings, outliers, categorical noise, and schema evolution across two sources. The benchmark compares no-op, static-rule, and MLP/fuzzy/GA methods over seeded trials.

This is a faithful, testable interpretation of the paper, not an exact reproduction: the paper does not specify neural architecture, fuzzy membership functions, GA fitness, or data splits. The MLP is a compact scikit-learn neural classifier rather than TensorFlow; it is a learned schema classifier, but not evidence that deep learning is necessary. Synthetic scores and local timing are not production-domain performance claims.

## Install

Python 3.10 or newer is required. From the repository root:

```powershell
# Only if .venv does not already exist:
python -m venv .venv

# Run these commands in the same PowerShell terminal:
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m schema_evolution --help
```

If the venv is already created, skip the creation line. Activation applies only to the current terminal; activate it again in each new PowerShell terminal. On macOS/Linux, use `source .venv/bin/activate` instead.

## Interactive mode

After installation and virtual-environment activation, start the guided terminal menu with no arguments:

```powershell
schema-evolution
```

Alternatively, run `python -m schema_evolution`. If PowerShell says `schema-evolution` is not recognized, the venv is probably not active in that terminal. Activate it there, or run `& ".\.venv\Scripts\schema-evolution.exe"` directly. Use the arrow keys to select **Process a JSONL data file**, **Generate synthetic test data**, **Run the synthetic benchmark**, or **Exit**, then press Enter. Prompts guide you through input/output paths and settings; the process option suggests the included sample and its three-record baseline when applicable. Processing and benchmark summaries are shown as terminal tables. Press Ctrl+C to cancel the current prompt.

The direct subcommands below remain available for scripts and automation.

## Run the tests

```powershell
pytest
ruff check src tests
```

## Run the end-to-end synthetic benchmark

```powershell
schema-evolution benchmark --trials 3 --records-per-source 100 --test-per-source 120 --seed 42 --output artifacts/benchmark-report.json
```

The JSON report includes repair precision/recall, missing-value completeness, schema transition detection/adaptability, accepted schema versions, and p50/p95/p99 processing latency for each method and trial. Completeness is the fraction of originally missing fields filled with a non-null value; numerical repairs count as correct when within 1.5 baseline robust-scale units of synthetic ground truth; schema adaptability is the fraction of expected source transitions accepted after confirmation. Test records are never used to fit the model or tune GA parameters.

## Process a JSONL stream

Records must be JSON objects. Set `_source` to a stable source identifier; omitted values use `default`. The first 50 records **per source** are reserved for baseline fitting and are not emitted as processed records.

```powershell
schema-evolution process input.jsonl --baseline-records-per-source 50 --output cleaned.jsonl --report report.json
```

Each JSONL output row contains the original `raw` record, a `cleaned` candidate, correction audit entries, drift evidence, and `quarantined`. Consumers should route quarantined rows for review; this local CLI does not connect to a broker or database.

### Try the included sample

The included sample has six records, with the first three used for baseline fitting:

```powershell
python -m schema_evolution process sample_stream.jsonl --baseline-records-per-source 3 --output prototype_output/cleaned.jsonl --report prototype_output/report.json
```

The command prints a summary. The cleaned audit records and full report are written to `prototype_output/`.

## Generate synthetic files

```powershell
schema-evolution generate-synthetic generated.jsonl --baseline-count 100 --test-count 120 --seed 42
schema-evolution process generated.jsonl --baseline-records-per-source 100 --output cleaned.jsonl --report process-report.json
```

This command writes a process-ready JSONL stream (the first `baseline-count` rows per source are warm-up) and a sibling `.ground-truth.jsonl` file aligned to test rows by source-local `test_index`. Do not expose ground-truth files to a production processing run.

## Synthetic Benchmark Results

The checked-in [benchmark report](artifacts/benchmark-report.json) summarizes **3 seeded trials** (seed 42), each with 100 baseline and 120 held-out test records per source across two synthetic sources.

| Method | Repair precision | Repair recall | Missing-value completeness | Schema adaptability | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proposed MLP + fuzzy + GA | **86.0%** | 57.2% | 66.7% | **100%** | 2.311 ms |
| Static rules | 84.8% | **63.6%** | **100%** | 0% | 2.374 ms |
| No-op | 0% | 0% | 0% | 0% | 0.0004 ms |

**What this indicates:** on this generated workload, the proposed pipeline detected and accepted both expected schema transitions in each trial and had slightly higher repair precision than the static baseline. Its conservative fuzzy abstention reduced fill rate and repair recall. The latest run's p95 latency was similar to the static baseline; timing varies with the local machine and load. This is a tradeoff, not a blanket performance win.

**Scope of the evidence:** these are local synthetic results, not validation on production data and not a reproduction of the paper's reported percentages or 12 ms latency. The benchmark's completeness is fill rate (non-null results for originally missing fields), not proof that imputed values are correct. Repair matches use a robust-scale tolerance; see [docs/architecture.md](docs/architecture.md) for metric definitions and limitations.

## Architecture and limitations

See [docs/architecture.md](docs/architecture.md) for component contracts, safety behavior, and operational boundaries. The current project is a local batch/JSONL implementation: it has bounded warm-up state but does not include Kafka, Flink, a schema registry, a REST service, durable checkpoints, or production monitoring. The paper's reported percentages and 12 ms result are not reproduced here.
