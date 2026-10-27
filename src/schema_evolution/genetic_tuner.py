"""DEAP-based GA calibration for fuzzy outlier sensitivity and confidence."""

from __future__ import annotations

import random
from typing import Any

from deap import algorithms, base, creator, tools

from schema_evolution.fuzzy_quality import FuzzyInference, FuzzyParameters


def _ensure_deap_types() -> None:
    if "SchemaEvolutionFitnessMax" not in creator.__dict__:
        creator.create("SchemaEvolutionFitnessMax", base.Fitness, weights=(1.0,))
    if "SchemaEvolutionIndividual" not in creator.__dict__:
        creator.create(
            "SchemaEvolutionIndividual",
            list,
            fitness=creator.__dict__["SchemaEvolutionFitnessMax"],
        )


class GeneticParameterTuner:
    """Tune two fuzzy parameters on synthetic labeled calibration examples."""

    def __init__(self, seed: int = 17, population_size: int = 16, generations: int = 10) -> None:
        if population_size < 4 or generations < 1:
            raise ValueError("GA requires population_size >= 4 and generations >= 1")
        self.seed = seed
        self.population_size = population_size
        self.generations = generations

    def tune(self, clean_values: list[float]) -> FuzzyParameters:
        if len(clean_values) < 8:
            return FuzzyParameters()
        import statistics

        center = statistics.median(clean_values)
        deviations = [abs(value - center) for value in clean_values]
        scale = statistics.median(deviations) * 1.4826
        if scale <= 1e-12:
            return FuzzyParameters()
        calibration: list[tuple[float, bool]] = [
            (abs(value - center) / scale, False) for value in clean_values
        ]
        for index, value in enumerate(clean_values):
            direction = 1.0 if index % 2 == 0 else -1.0
            corrupted = value + direction * 8.0 * scale
            calibration.append((abs(corrupted - center) / scale, True))
        fuzzy = FuzzyInference()
        scored_calibration = [
            (robust_z, label, fuzzy.outlier_confidence(robust_z, len(clean_values)))
            for robust_z, label in calibration
        ]

        def fitness(individual: list[float]) -> tuple[float]:
            sensitivity, confidence_threshold = individual
            true_positive = false_positive = true_negative = false_negative = 0
            for robust_z, is_corrupt, confidence in scored_calibration:
                predicted = robust_z >= sensitivity and confidence >= confidence_threshold
                if predicted and is_corrupt:
                    true_positive += 1
                elif predicted:
                    false_positive += 1
                elif is_corrupt:
                    false_negative += 1
                else:
                    true_negative += 1
            sensitivity_score = true_positive / max(1, true_positive + false_negative)
            specificity = true_negative / max(1, true_negative + false_positive)
            return ((sensitivity_score + specificity) / 2.0,)

        _ensure_deap_types()
        toolbox: Any = base.Toolbox()
        toolbox.register("sensitivity", random.uniform, 2.0, 7.0)
        toolbox.register("confidence", random.uniform, 0.55, 0.92)
        sensitivity_attribute = toolbox.__dict__["sensitivity"]
        confidence_attribute = toolbox.__dict__["confidence"]
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.__dict__["SchemaEvolutionIndividual"],
            (sensitivity_attribute, confidence_attribute),
            n=1,
        )
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.__dict__["individual"]
        )
        toolbox.register("evaluate", fitness)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def mate(first: Any, second: Any) -> tuple[Any, Any]:
            tools.cxBlend(first, second, alpha=0.25)
            self._bound(first)
            self._bound(second)
            return first, second

        def mutate(individual: Any) -> tuple[Any]:
            tools.mutGaussian(individual, mu=0.0, sigma=0.6, indpb=0.5)
            self._bound(individual)
            return (individual,)

        toolbox.register("mate", mate)
        toolbox.register("mutate", mutate)
        random_state = random.getstate()
        random.seed(self.seed)
        try:
            population = toolbox.__dict__["population"](n=self.population_size)
            hall_of_fame = tools.HallOfFame(1)
            algorithms.eaSimple(
                population,
                toolbox,
                cxpb=0.65,
                mutpb=0.3,
                ngen=self.generations,
                halloffame=hall_of_fame,
                verbose=False,
            )
        finally:
            random.setstate(random_state)
        best = hall_of_fame[0]
        return FuzzyParameters(
            outlier_sensitivity=float(best[0]), minimum_confidence=float(best[1])
        )

    @staticmethod
    def _bound(individual: list[float]) -> None:
        individual[0] = min(7.0, max(2.0, individual[0]))
        individual[1] = min(0.92, max(0.55, individual[1]))
