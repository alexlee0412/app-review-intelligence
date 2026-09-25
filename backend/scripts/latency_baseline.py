"""Run a fixed question set and record query-run identifiers for latency analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MANIFEST_SCHEMA_VERSION = "latency-baseline-v1"
DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("baseline_questions.json")
DEFAULT_OUTPUT_PATH = Path("latency-baseline-manifest.json")
EXPECTED_SINGLE_CALL_SHAPES = frozenset({"insufficiency", "trend_analysis"})


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the existing grounded-answer pipeline without changing it."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--repetitions", type=_positive_integer, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=_positive_integer)
    return parser


def _session_factory() -> Any:
    from app.core.db import SessionLocal

    return SessionLocal


def _answer_question(session: object, question: str) -> object:
    from app.services.ask_service import answer_question

    return answer_question(session, question)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_questions(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("question set must be a non-empty JSON array")

    questions: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each question entry must be an object")
        question_id = item.get("id")
        question = item.get("question")
        shape = item.get("shape")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (question_id, question, shape)
        ):
            raise ValueError("each question requires non-blank id, question, and shape")
        if question_id in identifiers:
            raise ValueError(f"duplicate question id: {question_id}")
        identifiers.add(question_id)
        questions.append(
            {
                "id": question_id,
                "question": question,
                "shape": shape,
            }
        )
    return questions, hashlib.sha256(raw).hexdigest()


def _execution_plan(
    questions: list[dict[str, str]], repetitions: int, limit: int | None
) -> list[tuple[dict[str, str], int]]:
    plan = [
        (question, repetition)
        for question in questions
        for repetition in range(1, repetitions + 1)
    ]
    return plan if limit is None else plan[:limit]


def _manifest(
    *,
    questions_path: Path,
    questions_hash: str,
    questions: list[dict[str, str]],
    repetitions: int,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "started_at": _now(),
        "finished_at": None,
        "question_set": {
            "source": questions_path.name,
            "sha256": questions_hash,
            "questions": [
                {"id": item["id"], "shape": item["shape"]} for item in questions
            ],
        },
        "repetitions": repetitions,
        "runs": [],
    }


def _estimated_model_calls(plan: list[tuple[dict[str, str], int]]) -> int:
    return sum(
        1 if question["shape"] in EXPECTED_SINGLE_CALL_SHAPES else 2
        for question, _ in plan
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(
    args: argparse.Namespace,
    *,
    answerer: Callable[[object, str], object] | None = None,
    session_factory: Callable[[], Any] | None = None,
) -> int:
    questions, questions_hash = _load_questions(args.questions)
    plan = _execution_plan(questions, args.repetitions, args.limit)
    print(f"Questions: {len(questions)}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Planned runs: {len(plan)}")
    print(f"Estimated model calls: {_estimated_model_calls(plan)}")
    print("Execution plan:")
    planned_by_question = Counter(question["id"] for question, _ in plan)
    for question in questions:
        planned = planned_by_question[question["id"]]
        if planned:
            print(f"- {question['id']} [{question['shape']}]: {planned} runs")
    if args.dry_run:
        return 0

    resolved_answerer = answerer or _answer_question
    resolved_session_factory = session_factory or _session_factory()
    manifest = _manifest(
        questions_path=args.questions,
        questions_hash=questions_hash,
        questions=questions,
        repetitions=args.repetitions,
    )
    _write_manifest(args.out, manifest)

    total = len(plan)
    for position, (question, repetition) in enumerate(plan, start=1):
        print(
            f"[{position}/{total}] {question['id']} repetition {repetition}",
            flush=True,
        )
        record: dict[str, Any] = {
            "question_id": question["id"],
            "repetition": repetition,
            "query_run_id": None,
            "ok": False,
        }
        try:
            with resolved_session_factory() as session:
                response = resolved_answerer(session, question["question"])
            record["query_run_id"] = str(response.query_run_id)
            record["ok"] = True
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            print(
                f"[{position}/{total}] failed ({type(exc).__name__}); continuing",
                file=sys.stderr,
                flush=True,
            )
        manifest["runs"].append(record)
        _write_manifest(args.out, manifest)

    manifest["finished_at"] = _now()
    _write_manifest(args.out, manifest)
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(
            f"fatal: unable to run latency baseline ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
