from schema_evolution.fuzzy_quality import FuzzyParameters, FuzzyQualityCorrector


def baseline_records():
    return [
        {"_source": "sensor", "reading": float(value), "state": "normal"} for value in range(10, 50)
    ]


def test_numeric_text_is_coerced_and_audited():
    cleaner = FuzzyQualityCorrector(baseline_records())
    cleaned, decisions = cleaner.correct(
        {"_source": "sensor", "reading": "24.5", "state": "normal"}
    )
    assert cleaned["reading"] == 24.5
    assert decisions[0].rule == "numeric_string_coercion"
    assert decisions[0].status == "corrected"


def test_missing_value_uses_baseline_and_preserves_audit_decision():
    cleaner = FuzzyQualityCorrector(baseline_records())
    cleaned, decisions = cleaner.correct({"_source": "sensor", "reading": None, "state": "normal"})
    assert cleaned["reading"] == 29.5
    assert decisions[0].status == "corrected"
    assert decisions[0].rule == "missing_value"


def test_low_support_abstains_without_mutating_the_raw_value():
    cleaner = FuzzyQualityCorrector(
        [{"value": 2.0}, {"value": 3.0}], parameters=FuzzyParameters(minimum_confidence=0.72)
    )
    cleaned, decisions = cleaner.correct({"value": None})
    assert cleaned["value"] is None
    assert decisions[0].status == "abstained"


def test_extreme_outlier_is_corrected_with_sufficient_support():
    cleaner = FuzzyQualityCorrector(baseline_records())
    cleaned, decisions = cleaner.correct(
        {"_source": "sensor", "reading": 1000.0, "state": "normal"}
    )
    decision = next(item for item in decisions if item.rule == "fuzzy_outlier_replacement")
    assert cleaned["reading"] == 29.5
    assert decision.status == "corrected"
