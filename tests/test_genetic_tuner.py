from schema_evolution.genetic_tuner import GeneticParameterTuner


def test_genetic_tuner_is_seeded_and_returns_bounded_parameters():
    values = [float(value) for value in range(10, 50)]
    tuner = GeneticParameterTuner(seed=5, population_size=8, generations=3)
    first = tuner.tune(values)
    second = tuner.tune(values)
    assert first == second
    assert 2.0 <= first.outlier_sensitivity <= 7.0
    assert 0.55 <= first.minimum_confidence <= 0.92
