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
    assert AnswerResponse.model_validate(json.loads(rendered)) == response
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
    for heading in ("Answer:", "Findings:", "Evidence:", "Limitations:", "Trace:"):
        assert heading in output
    assert "[observed]" in output
    assert "[E1] App One, 1★, 2026-07-01" in output
    assert '"Cannot cancel."' in output


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
