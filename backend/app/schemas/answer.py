"""Evidence and answer contracts for grounded question answering.

Two rules shape these schemas. Every review offered to the synthesis step carries a
stable identifier so a claim can be traced back to a specific row, and every number
lives in an aggregate computed by SQL rather than in prose, so no figure depends on
model arithmetic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.query_plan import QueryPlan
from app.schemas.search import AppliedFilters, ReviewEvidence

#: Below this many evidence rows, a question is answered as insufficient rather than
#: summarized. The dataset is small; confident prose over two reviews would mislead.
MIN_EVIDENCE_FOR_SYNTHESIS = 3

EVIDENCE_ID_PREFIX = "E"


def evidence_id_for(position: int) -> str:
    """Return the stable citation id for a 1-based evidence position."""
    if position < 1:
        raise ValueError("evidence positions are 1-based")
    return f"{EVIDENCE_ID_PREFIX}{position}"


class EvidenceItem(BaseModel):
    """A single cited review."""

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    review: ReviewEvidence
    # Verbatim from the stored review, only trimmed. Never rewritten or paraphrased.
    excerpt: str


class AppAggregate(BaseModel):
    """Deterministic per-app figures computed in SQL."""

    app_id: str
    app_name: str
    review_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    avg_rating: float | None = None
    rating_distribution: dict[int, int] = Field(default_factory=dict)
    oldest_review_at: datetime | None = None
    newest_review_at: datetime | None = None


class RetrievalTrace(BaseModel):
    """How the evidence was retrieved."""

    semantic_query: str | None = None
    top_k: int
    total_candidates: int = Field(ge=0)
    returned_evidence: int = Field(ge=0)
    embedding_provider: str | None = None
    embedding_is_production_grade: bool | None = None
    similarity_metric: str | None = None
    sql_template: str | None = None


class EvidenceBundle(BaseModel):
    """Everything the synthesis step is permitted to see."""

    query_plan: QueryPlan
    applied_filters: AppliedFilters
    aggregates: list[AppAggregate] = Field(default_factory=list)
    totals: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    total_candidates: int = Field(default=0, ge=0)
    limitations: list[str] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace

    @property
    def evidence_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence}

    @property
    def has_sufficient_evidence(self) -> bool:
        """Whether there is enough material to justify a synthesized answer."""
        return len(self.evidence) >= MIN_EVIDENCE_FOR_SYNTHESIS


class Finding(BaseModel):
    """One supported statement.

    ``kind`` forces the distinction between what was read, what was computed, and
    what is inference, so a reader can weigh each differently.
    """

    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    kind: Literal["observed", "computed", "interpretation"]

    @field_validator("evidence_ids")
    @classmethod
    def deduplicate(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class AnswerTrace(BaseModel):
    """Inspectable record of how the answer was produced."""

    intent: str
    applied_filters: AppliedFilters
    semantic_query: str | None = None
    total_candidates: int = 0
    evidence_count: int = 0
    aggregates_computed: list[str] = Field(default_factory=list)
    planner_model: str | None = None
    synthesizer_model: str | None = None
    llm_provider: str | None = None
    llm_is_production_grade: bool | None = None
    embedding_provider: str | None = None
    embedding_is_production_grade: bool | None = None
    synthesis_skipped: bool = False


class AnswerResponse(BaseModel):
    """A grounded answer with its supporting evidence and trace."""

    query_run_id: UUID
    question: str
    answer: str
    findings: list[Finding] = Field(default_factory=list)
    # Copied from the computed aggregates after synthesis. Never model-authored.
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace: AnswerTrace


class SynthesisOutput(BaseModel):
    """Raw synthesis result, before citation validation.

    Kept separate from :class:`AnswerResponse` so unverified model output is never
    mistaken for a checked answer.
    """

    answer: str
    findings: list[Finding] = Field(default_factory=list)
