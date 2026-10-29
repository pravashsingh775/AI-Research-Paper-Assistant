from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.app.core.database import get_session
from apps.api.app.main import app, current_user
from apps.api.app.models import User


def test_search_contract_rejects_short_queries() -> None:
    response = TestClient(app).post("/api/search", json={"topic": "a"})
    assert response.status_code == 422


def test_upload_requires_authentication() -> None:
    response = TestClient(app).post(
        "/api/papers/upload", files={"file": ("notes.txt", b"text", "text/plain")}
    )
    assert response.status_code == 401


def test_uploaded_paper_qa_requires_authentication() -> None:
    response = TestClient(app).post(
        "/api/qa",
        json={
            "document_id": "00000000-0000-0000-0000-000000000000",
            "question": "What dataset was used?",
        },
    )
    assert response.status_code == 401


def test_comparison_requires_authentication() -> None:
    response = TestClient(app).post(
        "/api/compare",
        json={
            "topic": "retrieval",
            "papers": [
                {
                    "id": "a",
                    "title": "Retrieval A",
                    "summary": "dense retrieval",
                    "year": 2024,
                },
                {
                    "id": "b",
                    "title": "Retrieval B",
                    "summary": "keyword retrieval",
                    "year": 2025,
                },
            ],
        },
    )
    assert response.status_code == 401


def test_comparison_is_derived_from_supplied_papers() -> None:
    user = User(id=uuid4(), email="test@example.com", password_hash="hash")
    app.dependency_overrides[current_user] = lambda: user
    try:
        response = TestClient(app).post(
            "/api/compare",
            json={
                "topic": "retrieval",
                "papers": [
                    {
                        "id": "a",
                        "title": "Retrieval A",
                        "summary": "dense retrieval",
                        "year": 2024,
                    },
                    {
                        "id": "b",
                        "title": "Retrieval B",
                        "summary": "keyword retrieval",
                        "year": 2025,
                    },
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["papers"][0]["paper"] == "Retrieval A"
    finally:
        app.dependency_overrides.pop(current_user, None)


def test_readiness_masks_dependency_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailedSession:
        async def execute(self, query: object) -> None:
            raise RuntimeError("database password must not be returned")

    class HealthyRedis:
        async def ping(self) -> bool:
            return True

        async def aclose(self) -> None:
            return None

    class HealthyStorage:
        async def put(self, key: str, data: bytes, content_type: str) -> None:
            return None

        async def exists(self, key: str) -> bool:
            return True

        async def delete(self, key: str) -> bool:
            return True

    monkeypatch.setattr(
        "redis.asyncio.from_url",
        lambda *args, **kwargs: HealthyRedis(),
    )
    monkeypatch.setattr("apps.api.app.main.get_storage", lambda: HealthyStorage())

    async def override_session():
        yield FailedSession()

    app.dependency_overrides[get_session] = override_session
    try:
        with TestClient(app) as client:
            response = client.get("/readiness")
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "status": "degraded",
            "checks": {"database": False, "redis": True, "storage": True},
        }
    }
    assert "password" not in response.text
