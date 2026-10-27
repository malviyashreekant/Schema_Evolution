"""Shared immutable data structures used by the pipeline components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DecisionStatus = Literal["corrected", "abstained", "observed"]


@dataclass(frozen=True)
class CorrectionDecision:
    field: str
    rule: str
    status: DecisionStatus
    confidence: float
    original: Any
    proposed: Any
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftEvent:
    source: str
    record_index: int
    previous_version: int
    added_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    type_changes: dict[str, dict[str, str]]
    model_probability: float
    candidate_observations: int
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["added_fields"] = list(self.added_fields)
        result["missing_fields"] = list(self.missing_fields)
        return result


@dataclass(frozen=True)
class PipelineResult:
    source: str
    record_index: int
    raw: dict[str, Any]
    cleaned: dict[str, Any]
    drift: DriftEvent | None
    corrections: tuple[CorrectionDecision, ...]
    quarantined: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "record_index": self.record_index,
            "raw": self.raw,
            "cleaned": self.cleaned,
            "schema_drift": self.drift.to_dict() if self.drift else None,
            "corrections": [decision.to_dict() for decision in self.corrections],
            "quarantined": self.quarantined,
        }
