"""Analyze persisted latency baseline runs without making model calls."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

STAGES = (
    "planner",
    "query_embedding",
    "analytics",
    "retrieval",
    "synthesis",
    "validation",
    "total",
)
MODEL_STAGES = ("planner", "synthesizer")
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report latency from a previously collected baseline manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _session_factory() -> Any:
    from app.core.db import SessionLocal

    return SessionLocal


def _load_rows(
    query_run_ids: list[str], session_factory: Callable[[], Any] | None = None
) -> dict[str, object]:
    if not query_run_ids:
        return {}

    from sqlalchemy import select

    from app.models import QueryRun

    parsed_ids = []
    for value in query_run_ids:
        try:
            parsed_ids.append(uuid.UUID(value))
        except (TypeError, ValueError, AttributeError):
            continue
    if not parsed_ids:
        return {}

    resolved_factory = session_factory or _session_factory()
    with resolved_factory() as session:
        rows = session.scalars(
            select(QueryRun).where(QueryRun.query_run_id.in_(parsed_ids))
        ).all()
    return {str(row.query_run_id): row for row in rows}


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = list(values)
    if not collected:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "n": len(collected),
        "mean": statistics.mean(collected),
        "median": statistics.median(collected),
        "min": min(collected),
        "max": max(collected),
        "stdev": statistics.stdev(collected) if len(collected) > 1 else None,
    }


def _variance_breakdown(
    values_by_question: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    within_stdevs = [
        statistics.stdev(values)
        for values in values_by_question.values()
        if len(values) > 1
    ]
    question_means = [
        statistics.mean(values) for values in values_by_question.values() if values
    ]
    within = {
        "questions_with_repeats": len(within_stdevs),
        "mean_stdev": statistics.mean(within_stdevs) if within_stdevs else None,
    }
    between = {
        "question_count": len(question_means),
        "stdev_of_means": (
            statistics.stdev(question_means) if len(question_means) > 1 else None
        ),
    }
    return within, between


def _stage_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        values_by_question: dict[str, list[float]] = defaultdict(list)
        values: list[float] = []
        paired_values: list[float] = []
        paired_totals: list[float] = []
        for record in records:
            value = _numeric(record["latency"].get(stage))
            if value is None:
                continue
            values.append(value)
            values_by_question[record["question_id"]].append(value)
            total = _numeric(record["latency"].get("total"))
            if total is not None:
                paired_values.append(value)
                paired_totals.append(total)
        summary = _statistics(values)
        paired_total_mean = (
            statistics.mean(paired_totals) if paired_totals else None
        )
        summary["share_of_mean_total"] = (
            statistics.mean(paired_values) / paired_total_mean
            if paired_values and paired_total_mean not in (None, 0)
            else None
        )
        summary["share_n"] = len(paired_values)
        within, between = _variance_breakdown(values_by_question)
        summary["within_question_spread"] = within
        summary["between_question_spread"] = between
        stages[stage] = summary
    return stages


def _model_usage(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    usage_report: dict[str, dict[str, Any]] = {}
    for stage in MODEL_STAGES:
        usage_items = []
        for record in records:
            model_usage = record["model_usage"]
            usage = model_usage.get(stage) if isinstance(model_usage, dict) else None
            if isinstance(usage, dict):
                usage_items.append(usage)
        providers = Counter(
            str(item["provider"])
            for item in usage_items
            if item.get("provider") is not None
        )
        models = Counter(
            str(item["model"])
            for item in usage_items
            if item.get("model") is not None
        )
        usage_report[stage] = {
            "runs_with_usage": len(usage_items),
            "runs_without_usage": len(records) - len(usage_items),
            "providers": dict(sorted(providers.items())),
            "models": dict(sorted(models.items())),
            "tokens": {
                field: _statistics(
                    value
                    for item in usage_items
                    if (value := _numeric(item.get(field))) is not None
                )
                for field in TOKEN_FIELDS
            },
        }
    return usage_report


def _cold_start(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, dict[int, float]] = defaultdict(dict)
    for record in records:
        total = _numeric(record["latency"].get("total"))
        repetition = record.get("repetition")
        if total is not None and isinstance(repetition, int):
            by_question[record["question_id"]][repetition] = total

    first_values: list[float] = []
    later_values: list[float] = []
    first_slower = 0
    for repetitions in by_question.values():
        first = repetitions.get(1)
        later = [value for index, value in repetitions.items() if index > 1]
        if first is None or not later:
            continue
        first_values.append(first)
        later_values.extend(later)
        if first > statistics.mean(later):
            first_slower += 1

    compared = len(first_values)
    systematically_slower = first_slower > compared / 2 if compared else None
    verdict_threshold = "strict majority (>50%)"
    if systematically_slower is None:
        note = "Cold-start comparison is unavailable."
    elif systematically_slower:
        note = (
            "Repetition #1 was slower for a majority of comparable questions "
            "(threshold: more than 50%)."
        )
    else:
        note = (
            "Repetition #1 was slower for no more than half of comparable questions "
            "(threshold: more than 50%)."
        )
    return {
        "questions_compared": compared,
        "questions_with_slower_first_run": first_slower,
        "first_repetition_mean": (
            statistics.mean(first_values) if first_values else None
        ),
        "later_repetitions_mean": (
            statistics.mean(later_values) if later_values else None
        ),
        "systematically_slower": systematically_slower,
        "verdict_threshold": verdict_threshold,
        "note": note,
    }


def build_report(
    manifest: dict[str, Any], rows_by_id: dict[str, object]
) -> dict[str, Any]:
    manifest_runs = manifest.get("runs")
    runs = manifest_runs if isinstance(manifest_runs, list) else []
    failed_runs = [
        {
            "question_id": item.get("question_id"),
            "repetition": item.get("repetition"),
            "error_type": item.get("error_type"),
        }
        for item in runs
        if isinstance(item, dict) and item.get("ok") is not True
    ]
    missing_ids: list[str] = []
    null_latency_ids: list[str] = []
    loaded_records: list[dict[str, Any]] = []
    timing_records: list[dict[str, Any]] = []
    successful_manifest_runs = 0
    for item in runs:
        if not isinstance(item, dict) or item.get("ok") is not True:
            continue
        successful_manifest_runs += 1
        query_run_id = item.get("query_run_id")
        if not isinstance(query_run_id, str) or query_run_id not in rows_by_id:
            missing_ids.append(str(query_run_id))
            continue
        row = rows_by_id[query_run_id]
        latency = getattr(row, "stage_latency_ms", None)
        record = {
            "question_id": str(item.get("question_id")),
            "repetition": item.get("repetition"),
            "query_run_id": query_run_id,
            "latency": latency if isinstance(latency, dict) else {},
            "model_usage": getattr(row, "model_usage", None),
        }
        loaded_records.append(record)
        if not isinstance(latency, dict):
            null_latency_ids.append(query_run_id)
            continue
        timing_records.append(record)

    skipped_synthesis = sum(
        1
        for record in timing_records
        if record["latency"].get("synthesis") is None
    )
    return {
        "manifest_schema_version": manifest.get("schema_version"),
        "run_counts": {
            "manifest_runs": len(runs),
            "successful_manifest_runs": successful_manifest_runs,
            "failed_manifest_runs": len(failed_runs),
            "loaded_database_rows": len(loaded_records),
            "analyzed_runs": len(timing_records),
            "missing_database_rows": len(missing_ids),
            "null_latency_rows": len(null_latency_ids),
        },
        "stages": _stage_statistics(timing_records),
        "synthesis_skipped_runs": skipped_synthesis,
        "model_usage": _model_usage(loaded_records),
        "failed_runs": failed_runs,
        "missing_query_run_ids": missing_ids,
        "null_stage_latency_query_run_ids": null_latency_ids,
        "cold_start": _cold_start(timing_records),
    }


def _format_number(value: object) -> str:
    return "undefined" if value is None else f"{float(value):.3f}"


def _print_report(report: dict[str, Any]) -> None:
    counts = report["run_counts"]
    print("Runs")
    print("----")
    print(
        f"Manifest: {counts['manifest_runs']} · "
        f"analyzed: {counts['analyzed_runs']} · "
        f"failed: {counts['failed_manifest_runs']} · "
        f"missing: {counts['missing_database_rows']} · "
        f"null latency: {counts['null_latency_rows']}"
    )
    print()
    print("Stage latency (milliseconds)")
    print("----------------------------")
    for stage in STAGES:
        stats = report["stages"][stage]
        within = stats["within_question_spread"]
        between = stats["between_question_spread"]
        print(
            f"{stage}: n={stats['n']} mean={_format_number(stats['mean'])} "
            f"median={_format_number(stats['median'])} "
            f"min={_format_number(stats['min'])} max={_format_number(stats['max'])} "
            f"stdev={_format_number(stats['stdev'])} "
            f"share={_format_number(stats['share_of_mean_total'])} "
            f"share_n={stats['share_n']}"
        )
        print(
            "  within-question mean stdev="
            f"{_format_number(within['mean_stdev'])} "
            f"({within['questions_with_repeats']} questions); "
            "between-question stdev of means="
            f"{_format_number(between['stdev_of_means'])} "
            f"({between['question_count']} questions)"
        )
    print(f"Synthesis skipped: {report['synthesis_skipped_runs']}")
    print()
    print("Token usage")
    print("-----------")
    for stage in MODEL_STAGES:
        usage = report["model_usage"][stage]
        print(
            f"{stage}: present={usage['runs_with_usage']} "
            f"absent={usage['runs_without_usage']}"
        )
        for field in TOKEN_FIELDS:
            stats = usage["tokens"][field]
            print(
                f"  {field}: n={stats['n']} mean={_format_number(stats['mean'])} "
                f"median={_format_number(stats['median'])} "
                f"min={_format_number(stats['min'])} "
                f"max={_format_number(stats['max'])} "
                f"stdev={_format_number(stats['stdev'])}"
            )
    print()
    print("Cold-start signal")
    print("-----------------")
    cold_start = report["cold_start"]
    print(
        f"Compared questions: {cold_start['questions_compared']} · "
        "slower first repetitions: "
        f"{cold_start['questions_with_slower_first_run']} of "
        f"{cold_start['questions_compared']} · "
        f"first mean: {_format_number(cold_start['first_repetition_mean'])} · "
        f"later mean: {_format_number(cold_start['later_repetitions_mean'])}"
    )
    print(cold_start["note"])
    if report["failed_runs"]:
        print()
        print("Failed runs")
        print("-----------")
        for failure in report["failed_runs"]:
            print(
                f"{failure['question_id']} repetition {failure['repetition']}: "
                f"{failure['error_type']}"
            )
    if report["missing_query_run_ids"]:
        print()
        print("Missing query_run_ids")
        print("---------------------")
        for query_run_id in report["missing_query_run_ids"]:
            print(query_run_id)
    if report["null_stage_latency_query_run_ids"]:
        print()
        print("Rows without stage latency")
        print("--------------------------")
        for query_run_id in report["null_stage_latency_query_run_ids"]:
            print(query_run_id)


def run(
    args: argparse.Namespace,
    *,
    row_loader: Callable[[list[str]], dict[str, object]] | None = None,
) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must contain a JSON object")
    manifest_runs = manifest.get("runs")
    runs = manifest_runs if isinstance(manifest_runs, list) else []
    query_run_ids = [
        item["query_run_id"]
        for item in runs
        if isinstance(item, dict)
        and item.get("ok") is True
        and isinstance(item.get("query_run_id"), str)
    ]
    rows = (row_loader or _load_rows)(query_run_ids)
    report = build_report(manifest, rows)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(
            f"fatal: unable to build latency report ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
