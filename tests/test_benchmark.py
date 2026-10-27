from schema_evolution.benchmark import run_benchmark


def test_seeded_synthetic_benchmark_runs_all_comparators():
    report = run_benchmark(trials=1, records_per_source=20, test_per_source=20, seed=4)
    methods = report["methods"]
    assert set(methods) == {"no_op", "static_rules", "proposed_mlp_fuzzy_ga"}
    assert methods["proposed_mlp_fuzzy_ga"]["schema_drift_detection_rate"] > 0.5
    assert methods["proposed_mlp_fuzzy_ga"]["completeness"] > methods["no_op"]["completeness"]
