"""Tests for ingestion command filesystem behavior."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.normalization import normalize_records
from scripts.ingest_apify import _parser, _resolve_rejects_dir, _write_rejects, run


def test_dry_run_creates_no_reject_files(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "id": "bad-review",
                    "appId": "app",
                    "score": 9,
                    "country": "US",
                    "date": "2026-01-01T00:00:00Z",
                    "text": "Invalid rating",
                }
            ]
        ),
        encoding="utf-8",
    )
    rejects_dir = tmp_path / "rejects"
    rejects_dir.mkdir()
    args = _parser().parse_args(
        [
            "--file",
            str(input_path),
            "--source",
            "test:dry-run",
            "--dry-run",
            "--rejects-dir",
            str(rejects_dir),
        ]
    )

    assert run(args) == 0
    assert list(rejects_dir.iterdir()) == []


def test_reject_artifact_contains_no_pii(tmp_path: Path) -> None:
    raw = {
        "id": "bad-review",
        "appId": "app",
        "score": 9,
        "country": "US",
        "date": "2026-01-01T00:00:00Z",
        "text": "Invalid rating",
        "userName": "private-name",
        "userUrl": "private-user-url",
        "reviewerId": "private-reviewer-id",
        "avatar": "private-avatar-url",
        "avatarUrl": "private-avatar-url",
    }
    result = normalize_records([raw], source="test:artifact")
    path = _write_rejects(tmp_path, result.rejects)

    assert path is not None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(artifact)
    for key in ("userName", "userUrl", "reviewerId", "avatar", "avatarUrl"):
        assert key not in serialized


def test_default_rejects_path_resolves_inside_project(tmp_path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args(
        ["--file", "input.json", "--source", "test:default-path", "--dry-run"]
    )

    resolved = _resolve_rejects_dir(args.rejects_dir).resolve()
    assert resolved.is_relative_to(project_root)
