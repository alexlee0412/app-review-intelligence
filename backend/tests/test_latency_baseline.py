from __future__ import annotations

import json
import uuid
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import latency_baseline


class SessionContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, *args: object) -> None:
        return None


def _questions(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "first",
                    "question": "Prompt text must not be persisted.",
                    "shape": "shape_one",
                },
                {
                    "id": "second",
                    "question": "Another private question.",
                    "shape": "shape_two",
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _args(questions: Path, output: Path, **overrides: object) -> Namespace:
    values = {
        "questions": questions,
        "repetitions": 1,
        "out": output,
        "dry_run": False,
        "limit": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_dry_run_makes_no_calls_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = _questions(tmp_path / "questions.json")
    output = tmp_path / "manifest.json"

    def unexpected(*args: object) -> object:
        raise AssertionError("dry-run invoked execution")

    result = latency_baseline.run(
        _args(questions, output, dry_run=True, repetitions=3),
        answerer=unexpected,
        session_factory=unexpected,
    )

    assert result == 0
    assert output.exists() is False
    rendered = capsys.readouterr().out
    assert "Planned runs: 6" in rendered
    assert "Estimated model calls: 12" in rendered
    assert "- first [shape_one]: 3 runs" in rendered
    assert "- second [shape_two]: 3 runs" in rendered


def test_manifest_survives_interruption_after_completed_run(tmp_path: Path) -> None:
    questions = _questions(tmp_path / "questions.json")
    output = tmp_path / "manifest.json"
    query_run_id = uuid.uuid4()
    calls = 0

    def answerer(session: object, question: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return SimpleNamespace(query_run_id=query_run_id)

    with pytest.raises(KeyboardInterrupt):
        latency_baseline.run(
            _args(questions, output),
            answerer=answerer,
            session_factory=SessionContext,
        )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["finished_at"] is None
    assert manifest["runs"] == [
        {
            "ok": True,
            "query_run_id": str(query_run_id),
            "question_id": "first",
            "repetition": 1,
        }
    ]


def test_failure_is_sanitized_and_sweep_continues(tmp_path: Path) -> None:
    questions = _questions(tmp_path / "questions.json")
    output = tmp_path / "manifest.json"
    query_run_id = uuid.uuid4()

    def answerer(session: object, question: str) -> object:
        if question.startswith("Prompt"):
            raise RuntimeError("sensitive exception detail with review and answer text")
        return SimpleNamespace(
            query_run_id=query_run_id,
            answer="answer text must not be persisted",
            evidence=["review text must not be persisted"],
        )

    assert (
        latency_baseline.run(
            _args(questions, output),
            answerer=answerer,
            session_factory=SessionContext,
        )
        == 0
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["finished_at"] is not None
    assert manifest["runs"] == [
        {
            "error_type": "RuntimeError",
            "ok": False,
            "query_run_id": None,
            "question_id": "first",
            "repetition": 1,
        },
        {
            "ok": True,
            "query_run_id": str(query_run_id),
            "question_id": "second",
            "repetition": 1,
        },
    ]
    serialized = output.read_text(encoding="utf-8")
    for forbidden in (
        "Prompt text must not be persisted",
        "Another private question",
        "sensitive exception detail",
        "answer text must not be persisted",
        "review text must not be persisted",
    ):
        assert forbidden not in serialized


def test_limit_caps_total_runs(tmp_path: Path) -> None:
    questions = _questions(tmp_path / "questions.json")
    output = tmp_path / "manifest.json"
    calls: list[str] = []

    def answerer(session: object, question: str) -> object:
        calls.append(question)
        return SimpleNamespace(query_run_id=uuid.uuid4())

    latency_baseline.run(
        _args(questions, output, repetitions=3, limit=2),
        answerer=answerer,
        session_factory=SessionContext,
    )

    assert calls == ["Prompt text must not be persisted."] * 2
    assert len(json.loads(output.read_text())["runs"]) == 2


def test_committed_question_set_covers_fixed_pipeline_shapes() -> None:
    questions, _ = latency_baseline._load_questions(
        latency_baseline.DEFAULT_QUESTIONS_PATH
    )
    assert {item["shape"] for item in questions} == {
        "app_specific_problems",
        "cross_app_theme",
        "pricing_paywall",
        "insufficiency",
        "trend_analysis",
        "broad_multi_app_summary",
    }
