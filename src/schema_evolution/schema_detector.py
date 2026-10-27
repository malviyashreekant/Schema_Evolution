"""Neural schema-change classification with explicit schema-version tracking."""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from schema_evolution.models import DriftEvent

_TYPE_NAMES = ("null", "boolean", "number", "string", "array", "object")


def infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "object"


def record_schema(record: dict[str, Any]) -> dict[str, str]:
    return {
        field: infer_type(value)
        for field, value in record.items()
        if field != "_source" and value is not None
    }


def consensus_schema(records: list[dict[str, Any]]) -> dict[str, str]:
    observed: dict[str, Counter[str]] = {}
    for record in records:
        for field, value in record.items():
            if field != "_source" and value is not None:
                observed.setdefault(field, Counter())[infer_type(value)] += 1
    if not observed:
        raise ValueError("Baseline records must contain at least one non-null data field")
    return {field: types.most_common(1)[0][0] for field, types in observed.items()}


class NeuralSchemaDetector:
    """MLP classifier trained from observed schemas and synthetic drift mutations."""

    def __init__(self, baseline_records: list[dict[str, Any]], seed: int = 17) -> None:
        self.seed = seed
        self.reference = consensus_schema(baseline_records)
        self.fields = tuple(sorted(self.reference))
        self.model = self._train(baseline_records)

    def _encode(self, schema: dict[str, str]) -> np.ndarray:
        vector: list[float] = []
        known_fields = set(self.fields)
        for field in self.fields:
            observed_type = schema.get(field)
            vector.append(float(observed_type is not None))
            vector.extend(float(observed_type == type_name) for type_name in _TYPE_NAMES)
        unknown_fields = set(schema) - known_fields
        vector.append(min(1.0, len(unknown_fields) / max(1, len(self.fields))))
        return np.asarray(vector, dtype=np.float32)

    def _mutations(self, schema: dict[str, str]) -> list[dict[str, str]]:
        variants: list[dict[str, str]] = []
        for field in schema:
            missing = dict(schema)
            del missing[field]
            variants.append(missing)
            changed = dict(schema)
            original_type = changed[field]
            changed[field] = "string" if original_type != "string" else "number"
            variants.append(changed)
        for index in range(4):
            added = dict(schema)
            added[f"__synthetic_new_{index}"] = "number" if index % 2 == 0 else "string"
            variants.append(added)
        return variants

    def _train(self, baseline_records: list[dict[str, Any]]) -> MLPClassifier:
        clean_schemas = [record_schema(record) for record in baseline_records]
        reference = consensus_schema(baseline_records)
        drift_schemas = self._mutations(reference)
        samples = clean_schemas + drift_schemas
        labels = [0] * len(clean_schemas) + [1] * len(drift_schemas)
        features = np.vstack(
            [self._encode_with_fields(schema, tuple(sorted(reference))) for schema in samples]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="lbfgs",
            alpha=0.01,
            max_iter=160,
            tol=1e-3,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            classifier.fit(features, labels)
        return classifier

    @staticmethod
    def _encode_with_fields(schema: dict[str, str], fields: tuple[str, ...]) -> np.ndarray:
        vector: list[float] = []
        for field in fields:
            observed_type = schema.get(field)
            vector.append(float(observed_type is not None))
            vector.extend(float(observed_type == type_name) for type_name in _TYPE_NAMES)
        vector.append(min(1.0, len(set(schema) - set(fields)) / max(1, len(fields))))
        return np.asarray(vector, dtype=np.float32)

    def probability(self, schema: dict[str, str]) -> float:
        encoded = self._encode(schema).reshape(1, -1)
        probabilities = self.model.predict_proba(encoded)[0]
        drift_index = list(self.model.classes_).index(1)
        return float(probabilities[drift_index])

    def rebase(self, records: list[dict[str, Any]]) -> None:
        self.reference = consensus_schema(records)
        self.fields = tuple(sorted(self.reference))
        self.model = self._train(records)


class SchemaEvolutionTracker:
    """Detect structural changes and accept a schema version after repeated evidence."""

    def __init__(
        self,
        source: str,
        baseline_records: list[dict[str, Any]],
        confirmations: int = 2,
        seed: int = 17,
    ) -> None:
        if confirmations < 1:
            raise ValueError("confirmations must be at least 1")
        self.source = source
        self.confirmations = confirmations
        self.version = 1
        self.expected = consensus_schema(baseline_records)
        self.detector = NeuralSchemaDetector(baseline_records, seed=seed)
        self.last_candidate: tuple[tuple[str, str], ...] | None = None
        self.candidate_observations = 0

    def observe(self, record: dict[str, Any], record_index: int) -> DriftEvent | None:
        actual = record_schema(record)
        present_fields = {field for field in record if field != "_source"}
        added = tuple(sorted(set(actual) - set(self.expected)))
        missing = tuple(sorted(set(self.expected) - present_fields))
        changed = {
            field: {"expected": self.expected[field], "observed": actual[field]}
            for field in set(actual) & set(self.expected)
            if actual[field] != self.expected[field]
        }
        if not (added or missing or changed):
            self.last_candidate = None
            self.candidate_observations = 0
            return None

        candidate_schema = dict(actual)
        for field in set(self.expected) & present_fields:
            candidate_schema.setdefault(field, self.expected[field])
        signature = tuple(sorted(candidate_schema.items()))
        if signature == self.last_candidate:
            self.candidate_observations += 1
        else:
            self.last_candidate = signature
            self.candidate_observations = 1
        probability = self.detector.probability(actual)
        accepted = self.candidate_observations >= self.confirmations
        event = DriftEvent(
            source=self.source,
            record_index=record_index,
            previous_version=self.version,
            added_fields=added,
            missing_fields=missing,
            type_changes=changed,
            model_probability=probability,
            candidate_observations=self.candidate_observations,
            accepted=accepted,
        )
        if accepted:
            self.version += 1
            self.expected = candidate_schema
            self.last_candidate = None
            self.candidate_observations = 0
        return event
