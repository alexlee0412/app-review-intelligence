from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.answer import AnswerResponse, AnswerTrace, EvidenceItem, Finding
from app.schemas.search import AppliedFilters, ReviewEvidence
from scripts import ask


class SessionContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, *args: object) -> None:
        return None


def _response() -> AnswerResponse:
    review = ReviewEvidence(
        review_id="review-one",
        app_id="app-one",
        app_name="App One",
        version="1.0",
        rating=1,
        country="US",
        created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        title=None,
        body="Cannot cancel.",
        similarity=0.9,
    )
    evidence = EvidenceItem(
        evidence_id="E1", review=review, excerpt="Cannot cancel."
    )
    filters = AppliedFilters(app_ids=["app-one"], countries=["US"], ratings=[1])
    return AnswerResponse(
        query_run_id=uuid.uuid4(),
        question="Why can users not cancel?",
        answer="A user reports a cancellation problem [E1].",
        findings=[
            Finding(
                claim="A user reports a cancellation problem.",
                evidence_ids=["E1"],
                kind="observed",
            )
        ],
        metrics={
            "totals": {
                "total_reviews": 10,
                "overall_avg_rating": 1.2857142857142858,
                "rating_distribution": {1: 5, 2: 2, 3: 1, 4: 0, 5: 2},
            },
            "apps": [
                {
                    "app_id": "app-one",
                    "app_name": "App One",
                    "review_count": 10,
                    "matched_count": 3,
                    "avg_rating": 1.2857142857142858,
                    "rating_distribution": {1: 5, 2: 2, 3: 1, 4: 0, 5: 2},
                }
            ],
        },
        evidence=[evidence],
        limitations=["Limited sample."],
        warnings=[],
        trace=AnswerTrace(
            intent="semantic_evidence",
            applied_filters=filters,
            semantic_query="cancel",
            total_candidates=1,
            evidence_count=1,
            synthesis_skipped=False,
        ),
    )


def test_json_output_validates_as_answer_response(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = _response()
    captured: dict[str, object] = {}

    def answer_question(session: object, question: str, *, overrides: object):
        captured["question"] = question
        captured["overrides"] = overrides
        return response

    monkeypatch.setattr(ask, "_session_factory", lambda: SessionContext)
    monkeypatch.setattr(ask, "answer_question", answer_question)
    args = ask._parser().parse_args(
        [
            "Why can users not cancel?",
            "--json",
            "--top-k",
            "4",
            "--app",
            "App One",
            "--app",
            "app-two",
            "--rating",
            "1,2,2",
        ]
    )

    assert ask.run(args) == 0
    rendered = capsys.readouterr().out
    decoded = json.loads(rendered)
    assert AnswerResponse.model_validate(decoded)
    assert decoded == json.loads(response.model_dump_json())
    assert decoded["metrics"]["totals"]["overall_avg_rating"] == 1.2857142857142858
    overrides = captured["overrides"]
    assert overrides.top_k == 4
    assert overrides.apps == ["App One", "app-two"]
    assert overrides.ratings == [1, 2]


def test_human_output_has_required_sections(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ask, "_session_factory", lambda: SessionContext)
    monkeypatch.setattr(ask, "answer_question", lambda *args, **kwargs: _response())
    args = ask._parser().parse_args(["Why can users not cancel?"])

    assert ask.run(args) == 0
    output = capsys.readouterr().out
    for heading in (
        "Answer:",
        "Metrics:",
        "Findings:",
        "Evidence:",
        "Limitations:",
        "Trace:",
    ):
        assert heading in output
    assert output.index("Metrics:") < output.index("Findings:")
    assert "Reviews analyzed: 10" in output
    assert "Average rating: 1.29" in output
    assert "1.2857142857142858" not in output
    assert "Rating distribution: 1★ 5 · 2★ 2 · 3★ 1 · 4★ 0 · 5★ 2" in output
    assert "[observed]" in output
    assert "[E1] App One, 1★, 2026-07-01" in output
    assert '"Cannot cancel."' in output


def test_human_output_renders_each_app_when_multiple_are_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = _response()
    metrics = dict(response.metrics)
    metrics["apps"] = [
        *metrics["apps"],
        {
            "app_id": "app-two",
            "app_name": "App Two",
            "review_count": 7,
            "matched_count": 2,
            "avg_rating": None,
            "rating_distribution": {},
        },
    ]
    response = response.model_copy(update={"metrics": metrics})
    monkeypatch.setattr(ask, "_session_factory", lambda: SessionContext)
    monkeypatch.setattr(ask, "answer_question", lambda *args, **kwargs: response)
    args = ask._parser().parse_args(["Why can users not cancel?"])

    assert ask.run(args) == 0
    output = capsys.readouterr().out
    assert (
        "App One: Reviews analyzed: 10 · Matched reviews: 3 · Average rating: 1.29"
        in output
    )
    assert "App Two: Reviews analyzed: 7 · Matched reviews: 2" in output


def test_human_output_omits_metrics_section_when_metrics_are_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = _response().model_copy(update={"metrics": {}})
    monkeypatch.setattr(ask, "_session_factory", lambda: SessionContext)
    monkeypatch.setattr(ask, "answer_question", lambda *args, **kwargs: response)

    assert ask.run(ask._parser().parse_args(["Why?"])) == 0
    assert "Metrics:" not in capsys.readouterr().out


def test_human_output_explains_withheld_narrative_without_findings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = _response().model_copy(
        update={
            "answer": (
                "The generated narrative was withheld because it contained figures "
                "that could not be grounded. Consult the validated findings below."
            ),
            "findings": [],
            "limitations": ["The answer narrative contained an unsupported figure."],
        }
    )
    monkeypatch.setattr(ask, "_session_factory", lambda: SessionContext)
    monkeypatch.setattr(ask, "answer_question", lambda *args, **kwargs: response)

    assert ask.run(ask._parser().parse_args(["Why?"])) == 0
    output = capsys.readouterr().out
    assert "generated narrative was withheld" in output
    assert "Nothing could be grounded." in output
    assert "Average rating: 1.29" in output
    assert "[E1] App One" in output


def test_main_reports_failure_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ask, "run", lambda args: (_ for _ in ()).throw(ValueError("bad input")))
    monkeypatch.setattr("sys.argv", ["ask.py", "question"])
    assert ask.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    # The class name is reported so a provider timeout is distinguishable from a
    # database or validation failure, but an arbitrary exception's message is not,
    # since it can embed a connection string.
    assert captured.err == "fatal: unable to answer question (ValueError)\n"
    assert "bad input" not in captured.err


def test_shell_wrapper_is_thin_relative_and_executable() -> None:
    wrapper = Path(__file__).resolve().parents[2] / "scripts/ask.sh"
    source = wrapper.read_text()
    assert os.access(wrapper, os.X_OK)
    assert "BASH_SOURCE" in source
    assert "backend/.venv/bin/python" in source
    assert "backend/scripts/ask.py" in source
    assert "/" + "Users/" not in source
