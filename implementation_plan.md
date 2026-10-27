# Execution Plan

## Goal

Build an evidence-backed pipeline that detects schema evolution and improves data quality without silently corrupting records. The repository now contains an installable Python package and local JSONL CLI implementing practical versions of the paper's MLP, fuzzy inference, and genetic optimization stages. Real-domain validation and durable streaming operations remain deployment prerequisites.

## Phase 1: End-to-end local package (implemented)

- Installable `src/` package, console entry point, and Windows/Ubuntu CI matrix.
- Per-source two-hidden-layer MLP schema classifier, explicit field/type diffs, and confirmation-based in-memory schema versions.
- Mamdani fuzzy confidence for numeric/categorical imputation and numeric outlier proposals; changes are auditable and low-confidence changes abstain.
- DEAP genetic tuning for sensitivity and confidence using generated calibration corruptions, separate from held-out benchmark records.
- Atomic JSONL/JSON artifacts with raw and cleaned records, correction provenance, drift evidence, quarantine flag, and latency summary.
- Seeded two-source synthetic generator and repeated comparisons against no-op and static-rule baselines.

**Acceptance checks:** editable install succeeds; tests, Ruff, process CLI, and benchmark CLI pass; no test data is used to fit models or tune GA parameters.

### Run locally

From the project directory, install into the existing environment, run tests and lint, then run the seeded benchmark:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m schema_evolution benchmark --trials 3 --records-per-source 100 --test-per-source 120 --seed 42 --output benchmark-report.json
```

Process a JSONL stream with `python -m schema_evolution process input.jsonl --baseline-records-per-source 50 --output cleaned.jsonl --report report.json`. The first 50 records per source are warm-up data and are not emitted. See [README.md](README.md) for synthetic generation and the complete CLI usage.

## Phase 2: Domain policy and review workflow (next)

- Define field-level required/optional status, allowed ranges/types, imputation eligibility, and source-specific thresholds with domain owners.
- Replace the output quarantine flag with durable dead-letter storage, access control, retention rules, and a human-review process.
- Persist schema versions and allow approval, rollback, and replay after a version change.
- Add correction lineage identifiers and explicit policy/model version IDs to each audit decision.

**Acceptance checks:** every proposed mutation can be traced to a versioned policy; a reviewer can approve/reject it; sensitive fields are never imputed without explicit authorization.

## Phase 3: Real-data offline evaluation (blocked on dataset access)

- Obtain representative, approved source data and document dataset versions, licenses, privacy constraints, and schema history.
- Replace synthetic MLP mutation training with labeled source-specific drift examples; split by time/source to avoid leakage.
- Compare against no-op, existing production rules, and the implemented static baseline on identical partitions.
- Pre-register correction quality, abstention rate, transition detection/acceptance, throughput, and p50/p95/p99 latency definitions.
- Run repeated trials and publish configuration, raw metrics, confidence intervals, and failure cases.

**Acceptance checks:** the benchmark can be reproduced from a versioned manifest; test data is never used for training/tuning; results meet domain-approved quality and false-correction limits.

## Phase 4: Calibrate and validate the learning components

- Validate whether the two-layer scikit-learn MLP is sufficient; compare with a TensorFlow sequence model only if labeled schema histories justify it.
- Review membership functions and fuzzy rules with domain experts; calibrate confidence to measured error costs and abstention targets.
- Extend GA fitness to the domain's explicit multi-objective cost, including false edits and latency; freeze parameters for final test runs.
- Add model artifacts, versioning, drift monitoring, and a validated fallback path.

**Acceptance checks:** each learned component has documented training data, reproducible parameters, ablation results, and a monitored fallback.

## Phase 5: Streaming deployment and operations

- Select broker/runtime based on delivery semantics, throughput, ordering, and recovery requirements; add bounded-memory state and checkpoints.
- Add idempotency, backpressure, retries, durable quarantine, metrics, tracing, secret/config management, and deployment automation.
- Load-test sustained and burst traffic, model startup, failures, restarts, replay, and schema transitions.
- Define service-level objectives and alerting before production rollout.

**Acceptance checks:** state remains bounded; restart/replay is deterministic; measured throughput and tail latency meet approved SLOs under fault tests.

## Suggested order of execution

1. Run the test suite and inspect the checked-in synthetic benchmark report.
2. Select one real source and approve its schema/correction policies.
3. Secure dataset access and reproduce the baseline metrics before tuning the model.
4. Implement durable review, rollback, and persistence controls.
5. Choose a streaming runtime and deploy only after load tests satisfy agreed SLOs.

## Open decisions

- Which real source format and source-specific schema should be supported first?
- Which fields may be imputed, and which must be rejected or quarantined?
- What is the acceptable false-positive rate and schema-change confirmation delay?
- What are the target throughput, latency percentiles, and deployment environment?
- Are the paper's original datasets, code, and baseline implementations available to reproduce its numbers?
- Which broker, deployment target, and operational SLOs are required?
