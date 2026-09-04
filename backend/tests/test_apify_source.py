"""Unit tests for local and HTTP Apify source adapters."""

from __future__ import annotations

import json

import httpx
import pytest

from app.services import apify_source
from app.services.apify_source import ApifyHTTPError, fetch_dataset, load_from_file


def test_load_from_file_detects_json_array(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text("  \n" + json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")
    assert list(load_from_file(path)) == [{"id": 1}, {"id": 2}]


def test_load_from_file_detects_jsonl(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('\n {"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    assert list(load_from_file(path)) == [{"id": 1}, {"id": 2}]


def _client_factory(transport: httpx.MockTransport):
    real_client = httpx.Client

    def factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    return factory


def test_fetch_dataset_retries_429_then_paginates(monkeypatch) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer " + "private-test-token"
        if len(calls) == 1:
            return httpx.Response(429, request=request)
        offset = int(request.url.params["offset"])
        payload = [{"id": 1}, {"id": 2}] if offset == 0 else [{"id": 3}]
        return httpx.Response(200, json=payload, request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(apify_source.httpx, "Client", _client_factory(transport))
    monkeypatch.setattr(apify_source.time, "sleep", lambda _: None)

    records = list(fetch_dataset("dataset/id", "private-test-token", page_size=2))
    assert records == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert len(calls) == 3
    assert all("private-test-token" not in str(call.url) for call in calls)


def test_http_error_is_typed_and_does_not_expose_token(monkeypatch) -> None:
    token = "private-test-token"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(apify_source.httpx, "Client", _client_factory(transport))

    with pytest.raises(ApifyHTTPError) as caught:
        list(fetch_dataset("dataset", token))
    assert caught.value.status_code == 401
    assert token not in str(caught.value)


def test_retry_is_bounded_to_five_attempts(monkeypatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(apify_source.httpx, "Client", _client_factory(transport))
    monkeypatch.setattr(apify_source.time, "sleep", lambda _: None)

    with pytest.raises(ApifyHTTPError) as caught:
        list(fetch_dataset("dataset", "private-test-token"))
    assert caught.value.status_code == 503
    assert calls == 5
