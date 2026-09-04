"""Operator-side adapters for reading raw Apify dataset records."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


class ApifySourceError(RuntimeError):
    """Base error for malformed input or failed Apify requests."""


class ApifyHTTPError(ApifySourceError):
    """A sanitized Apify HTTP failure."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Apify dataset request failed with HTTP status {status_code}")


def _require_object(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ApifySourceError(f"Apify record at {location} must be a JSON object")
    return value


def load_from_file(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSON array or JSONL file."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            character = handle.read(1)
            if not character:
                return
            if not character.isspace():
                first = character
                break
        handle.seek(0)

        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ApifySourceError("JSON input must contain an array of objects")
            for index, item in enumerate(payload, start=1):
                yield _require_object(item, location=f"array index {index - 1}")
            return

        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ApifySourceError(
                    f"Invalid JSONL record at line {line_number}"
                ) from exc
            yield _require_object(item, location=f"line {line_number}")


def fetch_dataset(
    dataset_id: str,
    token: str,
    *,
    page_size: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Yield all records from an Apify dataset using bounded page retries."""
    if page_size < 1:
        raise ValueError("page_size must be positive")

    url = (
        "https://api.apify.com/v2/datasets/"
        f"{quote(dataset_id, safe='')}/items"
    )
    headers = {"Authorization": "Bearer " + token}
    offset = 0

    with httpx.Client(timeout=30.0, follow_redirects=False) as client:
        while True:
            response: httpx.Response | None = None
            for attempt in range(5):
                try:
                    response = client.get(
                        url,
                        headers=headers,
                        params={
                            "clean": "true",
                            "offset": offset,
                            "limit": page_size,
                        },
                    )
                except httpx.HTTPError as exc:
                    raise ApifySourceError("Apify dataset request failed") from exc

                if response.status_code != 429 and response.status_code < 500:
                    break
                if attempt == 4:
                    raise ApifyHTTPError(response.status_code)
                time.sleep(min(0.5 * (2**attempt), 4.0))

            if response is None:
                raise ApifySourceError("Apify dataset request failed")
            if not 200 <= response.status_code < 300:
                raise ApifyHTTPError(response.status_code)

            try:
                payload = response.json()
            except ValueError as exc:
                raise ApifySourceError("Apify returned invalid JSON") from exc
            if not isinstance(payload, list):
                raise ApifySourceError("Apify dataset response must be a JSON array")

            for index, item in enumerate(payload):
                yield _require_object(item, location=f"offset {offset + index}")

            if len(payload) < page_size:
                return
            offset += len(payload)
