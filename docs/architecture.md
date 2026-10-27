# Architecture

## Data flow

1. The CLI scans the input JSONL once to retain a fixed warm-up sample for each source.
2. Each source gets its own MLP schema detector and quality profile. The GA is run during setup against generated calibration examples from that source's baseline.
3. A second pass streams records through per-source state. Structural differences are logged with model probability and field-level evidence.
4. A schema candidate is promoted only after the configured number of consecutive observations. Promotion updates the active structural profile without retraining the MLP on the record-processing path. It creates an in-memory version, not a durable schema registry update.
5. The fuzzy corrector proposes a repair. It mutates only the cleaned copy when its confidence meets the configured threshold. Raw input and all decisions remain in output.
6. Records with at least one abstained correction are marked `quarantined`; storage and review routing are left to the caller.

## Component details

### Schema model

`NeuralSchemaDetector` encodes field presence, observed type, and unseen-field ratio. A two-hidden-layer `MLPClassifier` distinguishes baseline signatures from generated mutations (missing/added fields and type changes). The exact diff is retained as an auditable guard because the paper provides no labeled schema-change corpus. Detection output includes the MLP's probability, candidate count, and whether the schema version was accepted.

The generated mutations are a bootstrap training strategy, not a substitute for source-specific labeled examples. The MLP remains anchored to the initial baseline, while the confirmed structural profile advances by version. Evaluate false alarms and detection delay against real schemas before production use.

### Fuzzy quality model

`FuzzyInference` uses Mamdani membership functions and centroid defuzzification. Numeric outlier confidence combines robust deviation and baseline support. Imputation confidence combines support, numeric stability, and categorical purity. The correction policy is conservative: low confidence means abstain and preserve the original value.

Numeric profiles are immutable after warm-up. Numeric values are coerced only when they parse to finite numbers. Outliers are proposed for median replacement; categorical values with the expected primitive type are not rewritten merely because a new label appears.

### Genetic tuner

The DEAP chromosome contains outlier sensitivity and minimum confidence. Fitness is balanced sensitivity/specificity on synthetic clean values plus known injected outliers. Parameters are tuned once per source at startup and are not learned from held-out benchmark records.

## Safety and operational boundaries

- Raw input is always included in each processed output row.
- The package never writes to input files; output files are replaced atomically after successful processing.
- Baseline examples, GA population, per-source profiles, and schema state reside in memory. Runtime state is not checkpointed across restarts.
- A source with fewer than the configured baseline records is rejected before processing.
- An unseen source encountered after warm-up is an error rather than silently borrowing another source's profile.
- The output `quarantined` flag is advisory; a production deployment must implement durable dead-letter storage, access control, retention, and human review.
- Synthetic benchmark latency excludes model fitting and is not a capacity estimate. Use representative data, realistic throughput, repeated load tests, and tail-latency SLOs before deployment.

## Synthetic benchmark metrics

- **Repair recall:** correctly repaired corrupted fields divided by all corrupted fields. A numerical result is counted as correct when it is within 1.5 robust-scale units of the generated clean value.
- **Repair precision:** correctly repaired fields divided by all fields the method changed.
- **Completeness:** originally missing fields made non-null divided by all originally missing fields; this measures fill rate, not imputation accuracy.
- **Schema drift detection:** source transitions with at least one drift event divided by expected source transitions.
- **Schema adaptability:** expected source transitions accepted after confirmation divided by expected source transitions.
- **Latency:** local time spent in `process` per test record; excludes baseline fitting and report serialization.

These are synthetic operational metrics, not direct reproductions of the paper's metric protocol.
