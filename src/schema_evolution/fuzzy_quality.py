"""Mamdani fuzzy confidence and conservative, auditable field correction."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import skfuzzy as fuzz

from schema_evolution.models import CorrectionDecision
from schema_evolution.schema_detector import infer_type


@dataclass(frozen=True)
class FuzzyParameters:
    outlier_sensitivity: float = 3.5
    minimum_confidence: float = 0.65


@dataclass
class FieldProfile:
    type_name: str
    numeric_values: list[float]
    categories: Counter[str]

    @property
    def median(self) -> float | None:
        return statistics.median(self.numeric_values) if self.numeric_values else None

    @property
    def robust_scale(self) -> float:
        if len(self.numeric_values) < 2:
            return 0.0
        median = statistics.median(self.numeric_values)
        deviations = [abs(value - median) for value in self.numeric_values]
        mad = statistics.median(deviations)
        if mad:
            return 1.4826 * mad
        ordered = sorted(self.numeric_values)
        lower = ordered[(len(ordered) - 1) // 4]
        upper = ordered[(3 * (len(ordered) - 1)) // 4]
        return (upper - lower) / 1.349

    @property
    def category_purity(self) -> float:
        count = sum(self.categories.values())
        return self.categories.most_common(1)[0][1] / count if count else 0.0


class FuzzyInference:
    """Small Mamdani system with explicit membership functions and centroid output."""

    def __init__(self) -> None:
        self.confidence_axis = np.linspace(0.0, 1.0, 101)
        self.low_output = fuzz.trapmf(self.confidence_axis, [0.0, 0.0, 0.18, 0.48])
        self.medium_output = fuzz.trimf(self.confidence_axis, [0.25, 0.55, 0.85])
        self.high_output = fuzz.trapmf(self.confidence_axis, [0.58, 0.82, 1.0, 1.0])

    def _centroid(self, rules: list[tuple[float, np.ndarray]]) -> float:
        aggregate = np.zeros_like(self.confidence_axis)
        for activation, output_set in rules:
            aggregate = np.fmax(aggregate, np.fmin(activation, output_set))
        if not np.any(aggregate):
            return 0.0
        return float(fuzz.defuzz(self.confidence_axis, aggregate, "centroid"))

    def outlier_confidence(self, robust_z: float, support: int) -> float:
        z_axis = np.linspace(0.0, 12.0, 241)
        support_axis = np.linspace(0.0, 1.0, 101)
        z = min(12.0, max(0.0, robust_z))
        support_ratio = min(1.0, support / 20.0)
        normal = fuzz.interp_membership(z_axis, fuzz.trapmf(z_axis, [0, 0, 1.5, 3]), z)
        elevated = fuzz.interp_membership(z_axis, fuzz.trimf(z_axis, [1.5, 4, 7]), z)
        extreme = fuzz.interp_membership(z_axis, fuzz.trapmf(z_axis, [4, 7, 12, 12]), z)
        adequate = fuzz.interp_membership(
            support_axis, fuzz.trapmf(support_axis, [0.2, 0.55, 1, 1]), support_ratio
        )
        limited = max(0.0, 1.0 - adequate)
        return self._centroid(
            [
                (normal, self.low_output),
                (min(elevated, adequate), self.medium_output),
                (min(extreme, adequate), self.high_output),
                (min(extreme, limited), self.medium_output),
            ]
        )

    def imputation_confidence(self, support: int, stability: float, purity: float = 1.0) -> float:
        support_axis = np.linspace(0.0, 1.0, 101)
        adequate = fuzz.interp_membership(
            support_axis,
            fuzz.trapmf(support_axis, [0.2, 0.65, 1, 1]),
            min(1.0, support / 20.0),
        )
        stable = min(1.0, max(0.0, stability))
        high_purity = min(1.0, max(0.0, purity))
        high_rule = min(adequate, stable, high_purity)
        medium_rule = max(min(adequate, stable), min(adequate, high_purity))
        low_rule = min(1.0 - adequate, 1.0 - stable, 1.0 - high_purity)
        return self._centroid(
            [
                (low_rule * 0.35, self.low_output),
                (medium_rule, self.medium_output),
                (high_rule, self.high_output),
            ]
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (float, int)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def build_profiles(records: list[dict[str, Any]]) -> dict[str, FieldProfile]:
    observations: dict[str, list[Any]] = {}
    for record in records:
        for field, value in record.items():
            if field != "_source" and value is not None:
                observations.setdefault(field, []).append(value)
    profiles: dict[str, FieldProfile] = {}
    for field, values in observations.items():
        numeric_values = [number for value in values if (number := _number(value)) is not None]
        if numeric_values and len(numeric_values) >= 0.8 * len(values):
            profiles[field] = FieldProfile("number", numeric_values, Counter())
        else:
            categories = Counter(str(value) for value in values)
            most_common_type = Counter(infer_type(value) for value in values).most_common(1)[0][0]
            profiles[field] = FieldProfile(most_common_type, [], categories)
    return profiles


class FuzzyQualityCorrector:
    def __init__(
        self,
        baseline_records: list[dict[str, Any]],
        parameters: FuzzyParameters | None = None,
        abstain_below: float | None = None,
    ) -> None:
        if not baseline_records:
            raise ValueError("Quality correction requires baseline records")
        self.profiles = build_profiles(baseline_records)
        self.parameters = parameters or FuzzyParameters()
        self.abstain_below = (
            self.parameters.minimum_confidence if abstain_below is None else abstain_below
        )
        self.fuzzy = FuzzyInference()

    def correct(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any], tuple[CorrectionDecision, ...]]:
        cleaned = dict(record)
        decisions: list[CorrectionDecision] = []
        for field, profile in self.profiles.items():
            original = record.get(field)
            if original is None:
                self._impute(cleaned, decisions, field, original, profile, "missing_value")
                continue
            if profile.type_name == "number":
                self._correct_numeric(cleaned, decisions, field, original, profile)
            elif infer_type(original) != profile.type_name and profile.categories:
                self._impute(cleaned, decisions, field, original, profile, "type_mismatch")
        return cleaned, tuple(decisions)

    def _impute(
        self,
        cleaned: dict[str, Any],
        decisions: list[CorrectionDecision],
        field: str,
        original: Any,
        profile: FieldProfile,
        rule: str,
    ) -> None:
        if profile.type_name == "number":
            proposed = profile.median
            scale = profile.robust_scale
            median = abs(proposed or 0.0)
            stability = 1.0 / (1.0 + scale / max(1.0, median)) if proposed is not None else 0.0
            confidence = self.fuzzy.imputation_confidence(len(profile.numeric_values), stability)
        elif profile.categories:
            proposed = profile.categories.most_common(1)[0][0]
            confidence = self.fuzzy.imputation_confidence(
                sum(profile.categories.values()),
                profile.category_purity,
                profile.category_purity,
            )
        else:
            return
        self._record_decision(cleaned, decisions, field, rule, original, proposed, confidence)

    def _correct_numeric(
        self,
        cleaned: dict[str, Any],
        decisions: list[CorrectionDecision],
        field: str,
        original: Any,
        profile: FieldProfile,
    ) -> None:
        number = _number(original)
        if number is None:
            self._impute(cleaned, decisions, field, original, profile, "invalid_numeric")
            return
        if isinstance(original, str):
            cleaned[field] = number
            decisions.append(
                CorrectionDecision(
                    field,
                    "numeric_string_coercion",
                    "corrected",
                    1.0,
                    original,
                    number,
                    "String parses as a finite number.",
                )
            )
        scale = profile.robust_scale
        median = profile.median
        if not scale or median is None:
            return
        robust_z = abs(number - median) / scale
        if robust_z < self.parameters.outlier_sensitivity:
            return
        confidence = self.fuzzy.outlier_confidence(robust_z, len(profile.numeric_values))
        self._record_decision(
            cleaned, decisions, field, "fuzzy_outlier_replacement", original, median, confidence
        )

    def _record_decision(
        self,
        cleaned: dict[str, Any],
        decisions: list[CorrectionDecision],
        field: str,
        rule: str,
        original: Any,
        proposed: Any,
        confidence: float,
    ) -> None:
        accepted = proposed is not None and confidence >= self.abstain_below
        if accepted:
            cleaned[field] = proposed
        decisions.append(
            CorrectionDecision(
                field=field,
                rule=rule,
                status="corrected" if accepted else "abstained",
                confidence=round(confidence, 6),
                original=original,
                proposed=proposed,
                explanation=(
                    "Fuzzy confidence met the configured threshold."
                    if accepted
                    else "Correction was withheld because confidence was below threshold."
                ),
            )
        )
