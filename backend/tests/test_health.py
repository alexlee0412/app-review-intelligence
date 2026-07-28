from fastapi.testclient import TestClient

from app.core.db import check_database_connection
from app.main import app


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client: TestClient) -> None:
    body = client.get("/health").json()
    assert set(body.keys()) == {"status", "service", "environment"}


def test_health_service_name(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["service"] == "App Review Intelligence API"


def test_health_environment_field_present(client: TestClient) -> None:
    body = client.get("/health").json()
    assert "environment" in body


def test_health_does_not_require_database(client: TestClient) -> None:
    def _fail_if_called() -> bool:
        raise AssertionError("GET /health must not depend on the database")

    app.dependency_overrides[check_database_connection] = _fail_if_called

    response = client.get("/health")

    assert response.status_code == 200


def test_health_db_reachable(client: TestClient) -> None:
    app.dependency_overrides[check_database_connection] = lambda: True

    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "reachable"}


def test_health_db_unreachable_returns_503(client: TestClient) -> None:
    app.dependency_overrides[check_database_connection] = lambda: False

    response = client.get("/health/db")

    assert response.status_code == 503


def test_health_db_failure_does_not_expose_credentials(client: TestClient) -> None:
    app.dependency_overrides[check_database_connection] = lambda: False

    response = client.get("/health/db")

    body_text = response.text.lower()
    assert "postgres" not in body_text
    assert "password" not in body_text
    assert "@" not in body_text
    assert "app_review" not in body_text
