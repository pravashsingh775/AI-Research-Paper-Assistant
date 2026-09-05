from uuid import uuid4
from fastapi.testclient import TestClient

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
