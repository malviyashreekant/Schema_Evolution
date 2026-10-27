"""Per-source orchestration for schema detection and quality correction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from schema_evolution.fuzzy_quality import FuzzyParameters, FuzzyQualityCorrector
from schema_evolution.genetic_tuner import GeneticParameterTuner
from schema_evolution.models import PipelineResult
from schema_evolution.schema_detector import SchemaEvolutionTracker


def source_name(record: dict[str, Any]) -> str:
    value = record.get("_source", "default")
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError("_source must be a JSON scalar")
    return str(value)


class PipelineEngine:
    """Stateful processor trained from a bounded baseline for each source."""

    def __init__(
        self,
        trackers: dict[str, SchemaEvolutionTracker],
        correctors: dict[str, FuzzyQualityCorrector],
    ) -> None:
        self.trackers = trackers
        self.correctors = correctors
        self.record_counts: defaultdict[str, int] = defaultdict(int)

    @classmethod
    def fit(
        cls,
        baseline_records: list[dict[str, Any]],
        *,
        seed: int = 17,
        confirmations: int = 2,
        tune_parameters: bool = True,
        fuzzy_abstention: bool = True,
        detect_schema: bool = True,
    ) -> PipelineEngine:
        if not baseline_records:
            raise ValueError("At least one baseline record is required")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in baseline_records:
            grouped[source_name(record)].append(record)
        trackers: dict[str, SchemaEvolutionTracker] = {}
        correctors: dict[str, FuzzyQualityCorrector] = {}
        for source, records in grouped.items():
            parameters = FuzzyParameters()
            if tune_parameters:
                numeric_values = []
                for record in records:
                    for field, value in record.items():
                        if field == "_source" or isinstance(value, bool) or value is None:
                            continue
                        try:
                            numeric_values.append(float(value))
                        except (TypeError, ValueError):
                            continue
                parameters = GeneticParameterTuner(seed=seed).tune(numeric_values)
            trackers[source] = (
                SchemaEvolutionTracker(source, records, confirmations=confirmations, seed=seed)
                if detect_schema
                else None
            )
            correctors[source] = FuzzyQualityCorrector(
                records,
                parameters=parameters,
                abstain_below=parameters.minimum_confidence if fuzzy_abstention else 0.0,
            )
        return cls(trackers, correctors)

    def process(self, record: dict[str, Any]) -> PipelineResult:
        source = source_name(record)
        if source not in self.correctors:
            raise ValueError(f"No baseline profile is available for source {source!r}")
        self.record_counts[source] += 1
        index = self.record_counts[source]
        tracker = self.trackers.get(source)
        drift = tracker.observe(record, index) if tracker else None
        cleaned, decisions = self.correctors[source].correct(record)
        quarantined = any(decision.status == "abstained" for decision in decisions)
        return PipelineResult(
            source=source,
            record_index=index,
            raw=dict(record),
            cleaned=cleaned,
            drift=drift,
            corrections=decisions,
            quarantined=quarantined,
        )
