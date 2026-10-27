from schema_evolution.schema_detector import SchemaEvolutionTracker


def test_schema_change_is_reported_then_accepted_after_confirmation():
    baseline = [
        {"_source": "weather", "temperature": 18.0, "status": "ok"},
        {"_source": "weather", "temperature": 20.0, "status": "ok"},
        {"_source": "weather", "temperature": 22.0, "status": "ok"},
    ]
    tracker = SchemaEvolutionTracker("weather", baseline, confirmations=2)
    evolved = {"_source": "weather", "temperature": 21.0, "status": "ok", "humidity": 48.0}

    candidate = tracker.observe(evolved, 1)
    accepted = tracker.observe(evolved, 2)

    assert candidate is not None
    assert candidate.added_fields == ("humidity",)
    assert candidate.candidate_observations == 1
    assert not candidate.accepted
    assert accepted is not None and accepted.accepted
    assert tracker.version == 2
    assert tracker.observe(evolved, 3) is None


def test_null_value_is_quality_data_not_a_structural_schema_change():
    baseline = [{"value": 1.0}, {"value": 2.0}]
    tracker = SchemaEvolutionTracker("default", baseline)
    assert tracker.observe({"value": None}, 1) is None
    assert tracker.observe({}, 2) is not None
