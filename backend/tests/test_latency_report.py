from __future__ import annotations

import json
import math
import uuid
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from scripts import latency_report


def _run(question_id: str, repetition: int, query_run_id: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "repetition": repetition,
        "query_run_id": query_run_id,
        "ok": True,
    }


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    ids = [str(uuid.uuid4()) for _ in range(4)]
    manifest = {
        "schema_version": "latency-baseline-v1",
        "runs": [
            _run("q1", 1, ids[0]),
            _run("q1", 2, ids[1]),
            _run("q2", 1, ids[2]),
            _run("q2", 2, ids[3]),
        ],
    }
    latencies = [
        {"planner": 10, "synthesis": 50, "validation": 1, "total": 100},
        {"planner": 12, "synthesis": None, "validation": None, "total": 102},
        {"planner": 30, "synthesis": 150, "validation": None, "total": 300},
        {"planner": 32, "synthesis": 151, "validation": None, "total": 302},
    ]
    rows = {
        query_run_id: SimpleNamespace(
            stage_latency_ms=latency,
            model_usage=(
                {
                    "planner": {
                        "provider": "openai",
                        "model": "planner-model",
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25,
                        "cached_input_tokens": None,
                    },
                    "synthesizer": None,
                }
                if index == 0
                else None
            ),
        )
        for index, (query_run_id, latency) in enumerate(zip(ids, latencies))
    }
    return manifest, rows


def test_statistics_and_variance_sources_are_reported_correctly() -> None:
    manifest, rows = _fixture()
    report = latency_report.build_report(manifest, rows)
    planner = report["stages"]["planner"]

    assert planner["n"] == 4
    assert planner["mean"] == 21
    assert planner["median"] == 21
    assert planner["min"] == 10
    assert planner["max"] == 32
    assert math.isclose(planner["stdev"], 11.604596, rel_tol=1e-6)
    within = planner["within_question_spread"]
    between = planner["between_question_spread"]
    assert within["questions_with_repeats"] == 2
    assert math.isclose(within["mean_stdev"], math.sqrt(2), rel_tol=1e-9)
    assert between["question_count"] == 2
    assert math.isclose(between["stdev_of_means"], math.sqrt(200), rel_tol=1e-9)
    assert math.isclose(planner["share_of_mean_total"], 21 / 201)
    assert planner["share_n"] == 4


def test_skipped_synthesis_and_single_value_stdev_are_explicit() -> None:
    manifest, rows = _fixture()
    report = latency_report.build_report(manifest, rows)

    assert report["synthesis_skipped_runs"] == 1
    assert report["stages"]["synthesis"]["n"] == 3
    assert report["stages"]["validation"]["n"] == 1
    assert report["stages"]["validation"]["stdev"] is None
    assert report["cold_start"] == {
        "questions_compared": 2,
        "questions_with_slower_first_run": 0,
        "first_repetition_mean": 200,
        "later_repetitions_mean": 202,
        "systematically_slower": False,
        "verdict_threshold": "strict majority (>50%)",
        "note": (
            "Repetition #1 was slower for no more than half of comparable questions "
            "(threshold: more than 50%)."
        ),
    }


def test_absent_usage_is_counted_without_fabricating_zeros() -> None:
    manifest, rows = _fixture()
    report = latency_report.build_report(manifest, rows)

    planner = report["model_usage"]["planner"]
    synthesizer = report["model_usage"]["synthesizer"]
    assert planner["runs_with_usage"] == 1
    assert planner["runs_without_usage"] == 3
    assert planner["tokens"]["input_tokens"]["n"] == 1
    assert planner["tokens"]["input_tokens"]["mean"] == 20
    assert planner["tokens"]["cached_input_tokens"]["n"] == 0
    assert synthesizer["runs_with_usage"] == 0
    assert synthesizer["runs_without_usage"] == 4


def test_missing_null_and_failed_runs_do_not_crash() -> None:
    present_id = str(uuid.uuid4())
    null_id = str(uuid.uuid4())
    missing_id = str(uuid.uuid4())
    manifest = {
        "schema_version": "latency-baseline-v1",
        "runs": [
            _run("present", 1, present_id),
            _run("null", 1, null_id),
            _run("missing", 1, missing_id),
            {
                "question_id": "failed",
                "repetition": 1,
                "query_run_id": None,
                "ok": False,
                "error_type": "RuntimeError",
            },
        ],
    }
    rows = {
        present_id: SimpleNamespace(
            stage_latency_ms={"total": 10}, model_usage=None
        ),
        null_id: SimpleNamespace(stage_latency_ms=None, model_usage=None),
    }

    report = latency_report.build_report(manifest, rows)

    assert report["run_counts"] == {
        "manifest_runs": 4,
        "successful_manifest_runs": 3,
        "failed_manifest_runs": 1,
        "loaded_database_rows": 2,
        "analyzed_runs": 1,
        "missing_database_rows": 1,
        "null_latency_rows": 1,
    }
    assert report["missing_query_run_ids"] == [missing_id]
    assert report["null_stage_latency_query_run_ids"] == [null_id]
    assert report["failed_runs"][0]["error_type"] == "RuntimeError"


def test_empty_manifest_has_undefined_statistics() -> None:
    report = latency_report.build_report(
        {"schema_version": "latency-baseline-v1", "runs": []}, {}
    )

    assert report["run_counts"]["analyzed_runs"] == 0
    assert report["stages"]["total"] == {
        "n": 0,
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
        "stdev": None,
        "share_of_mean_total": None,
        "share_n": 0,
        "within_question_spread": {
            "questions_with_repeats": 0,
            "mean_stdev": None,
        },
        "between_question_spread": {
            "question_count": 0,
            "stdev_of_means": None,
        },
    }
    assert report["cold_start"]["systematically_slower"] is None


def test_partial_stage_share_uses_paired_total_population() -> None:
    ids = [str(uuid.uuid4()) for _ in range(12)]
    manifest = {
        "schema_version": "latency-baseline-v1",
        "runs": [_run(f"q{index}", 1, value) for index, value in enumerate(ids)],
    }
    rows = {}
    for index, query_run_id in enumerate(ids):
        synthesis_present = index < 9
        rows[query_run_id] = SimpleNamespace(
            stage_latency_ms={
                "planner": 1,
                "query_embedding": 1,
                "analytics": 1,
                "retrieval": 1,
                "synthesis": 100 if synthesis_present else None,
                "validation": 1,
                "total": 110 if synthesis_present else 10,
            },
            model_usage=None,
        )

    report = latency_report.build_report(manifest, rows)
    synthesis = report["stages"]["synthesis"]

    assert synthesis["n"] == 9
    assert synthesis["share_n"] == 9
    assert math.isclose(synthesis["share_of_mean_total"], 100 / 110)
    assert synthesis["share_of_mean_total"] <= 1.0
    component_share = sum(
        report["stages"][stage]["share_of_mean_total"]
        for stage in latency_report.STAGES
        if stage != "total"
    )
    assert component_share <= 1.0


def _cold_start_records(
    *,
    question_count: int,
    slower_first_count: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(question_count):
        first = 20 if index < slower_first_count else 5
        records.extend(
            [
                {
                    "question_id": f"q{index}",
                    "repetition": 1,
                    "latency": {"total": first},
                    "model_usage": None,
                },
                {
                    "question_id": f"q{index}",
                    "repetition": 2,
                    "latency": {"total": 10},
                    "model_usage": None,
                },
            ]
        )
    return records


def test_clear_majority_of_slower_first_runs_is_reported() -> None:
    cold_start = latency_report._cold_start(
        _cold_start_records(question_count=12, slower_first_count=11)
    )

    assert cold_start["questions_with_slower_first_run"] == 11
    assert cold_start["questions_compared"] == 12
    assert cold_start["systematically_slower"] is True
    assert "not systematically slower" not in cold_start["note"]
    assert "more than 50%" in cold_start["note"]


def test_clear_minority_of_slower_first_runs_is_reported() -> None:
    cold_start = latency_report._cold_start(
        _cold_start_records(question_count=12, slower_first_count=3)
    )

    assert cold_start["questions_with_slower_first_run"] == 3
    assert cold_start["questions_compared"] == 12
    assert cold_start["systematically_slower"] is False
    assert "no more than half" in cold_start["note"]


def test_empty_manifest_run_does_not_load_database(
    tmp_path: Path, capsys
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "latency-baseline-v1", "runs": []})
    )
    captured: list[list[str]] = []

    def loader(values: list[str]) -> dict[str, object]:
        captured.append(values)
        return {}

    assert (
        latency_report.run(
            Namespace(manifest=manifest_path, as_json=True), row_loader=loader
        )
        == 0
    )
    assert captured == [[]]
    report = json.loads(capsys.readouterr().out)
    assert report["run_counts"]["manifest_runs"] == 0


def test_json_run_uses_only_manifest_ids(
    tmp_path: Path, capsys
) -> None:
    query_run_id = str(uuid.uuid4())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "latency-baseline-v1",
                "runs": [_run("q1", 1, query_run_id)],
            }
        )
    )
    captured_ids: list[str] = []

    def loader(values: list[str]) -> dict[str, object]:
        captured_ids.extend(values)
        return {
            query_run_id: SimpleNamespace(
                stage_latency_ms={"total": 10}, model_usage=None
            )
        }

    assert (
        latency_report.run(
            Namespace(manifest=manifest_path, as_json=True), row_loader=loader
        )
        == 0
    )
    assert captured_ids == [query_run_id]
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["run_counts"]["analyzed_runs"] == 1
