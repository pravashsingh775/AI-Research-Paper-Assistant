from __future__ import annotations

from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from apps.api.app.core.config import Settings
from apps.api.app.core.security import create_access_token
from apps.api.app.main import app, current_user
from apps.api.app.models import Paper, User


ANALYTICAL_PATHS = [
    "/api/compare",
    "/api/trends",
    "/api/research-gaps",
    "/api/research-ideas",
    "/api/proposals",
    "/api/similarity-map",
]


def test_analytical_tools_require_authentication() -> None:
    client = TestClient(app)
    body = {
        "topic": "Neural networks",
        "papers": [
            {"id": "a", "title": "Paper A", "summary": "Sample A"},
            {"id": "b", "title": "Paper B", "summary": "Sample B"},
        ],
    }
    for path in ANALYTICAL_PATHS:
        response = client.post(path, json=body)
        assert response.status_code == 401, (
            f"{path} did not reject unauthenticated access"
        )


def test_authenticated_user_can_access_analytical_tools() -> None:
    client = TestClient(app)
    user = User(id=uuid4(), email="alice@example.com", password_hash="hash")
    app.dependency_overrides[current_user] = lambda: user

    try:
        body = {
            "topic": "Neural networks",
            "papers": [
                {"id": "a", "title": "Paper A", "summary": "Sample A"},
                {"id": "b", "title": "Paper B", "summary": "Sample B"},
            ],
        }
        for path in ANALYTICAL_PATHS:
            response = client.post(path, json=body)
            assert response.status_code == 200, (
                f"{path} failed for authorized user: {response.text}"
            )
    finally:
        app.dependency_overrides.pop(current_user, None)


def test_production_guard_rejects_weak_or_default_secret() -> None:
    # Development allows default secret
    dev_settings = Settings(app_env="development", jwt_secret="change-me-in-production")
    dev_settings.check_production_guard()  # Should not raise

    # Production rejects default secret
    with pytest.raises(RuntimeError, match="FATAL: In production"):
        prod_settings = Settings(
            app_env="production", jwt_secret="change-me-in-production"
        )
        prod_settings.check_production_guard()

    # Production rejects short secret (<32 chars)
    with pytest.raises(RuntimeError, match="FATAL: In production"):
        short_settings = Settings(app_env="production", jwt_secret="short-key-123")
        short_settings.check_production_guard()

    # Production accepts strong secret (>=32 chars)
    secure_settings = Settings(
        app_env="production",
        jwt_secret="super-secure-production-jwt-key-minimum-32-chars-long",
    )
    secure_settings.check_production_guard()  # Should not raise
