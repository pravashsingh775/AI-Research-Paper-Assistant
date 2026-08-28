"""
Lightweight health and readiness endpoints for the AI Research Paper Assistant.

Design goals
------------
* /health answers liveness only.
* /health/ready answers application readiness using cheap state inspection.
* No model inference, FAISS search, PDF processing, LLM calls, embeddings,
  external network calls, or filesystem scans are performed here.
* Services are obtained from the application's existing state/lifecycle.
* Secrets, paths, exception internals, and implementation details are never
  returned to clients.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# ---------------------------------------------------------------------------
# Response contracts
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Minimal liveness response."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern=r"^healthy$")
    service: str


class ReadinessResponse(BaseModel):
    """Readiness response containing only safe component state."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern=r"^(ready|not_ready)$")
    service: str
    checks: dict[str, str]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_SERVICE_NAME = "AI Research Paper Assistant"

# app.state.health_required_services may be configured by main.py as:
#
#   app.state.health_required_services = {
#       "research_service": True,
#       "paper_indexing_service": True,
#       "rag_pipeline": True,
#   }
#
# If it is absent, health.py does NOT invent dependencies. It derives checks
# only from service objects that are actually present in app.state.
DEFAULT_KNOWN_SERVICES = (
    "research_service",
    "research_analyzer",
    "paper_indexing_service",
    "rag_pipeline",
    "research_search_service",
    "llm_client",
    "embedding_encoder",
)


def _safe_service_name(request: Request) -> str:
    """
    Read the application name from existing state/configuration if available.

    Only a safe string is returned. No environment/config secrets are exposed.
    """
    app_state = request.app.state

    for attribute in ("service_name", "app_name", "application_name"):
        value = getattr(app_state, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return DEFAULT_SERVICE_NAME


def _required_services(request: Request) -> dict[str, bool]:
    """
    Resolve readiness requirements without creating a second service registry.

    Preferred source:
        app.state.health_required_services

    Supported compatibility aliases:
        app.state.required_health_services
        app.state.required_services

    If none exists, no dependency is invented. Present service objects are
    reported, but they are not automatically made critical.
    """
    app_state = request.app.state

    for attribute in (
        "health_required_services",
        "required_health_services",
        "required_services",
    ):
        configured = getattr(app_state, attribute, None)

        if isinstance(configured, Mapping):
            result: dict[str, bool] = {}

            for name, required in configured.items():
                if (
                    isinstance(name, str)
                    and name.strip()
                    and isinstance(required, bool)
                ):
                    result[name.strip()] = required

            return result

    # No explicit registry: inspect only service names that actually exist.
    result = {}
    for name in DEFAULT_KNOWN_SERVICES:
        if hasattr(app_state, name):
            result[name] = False

    return result


def _lifecycle_ready(request: Request) -> Optional[bool]:
    """
    Read an optional application-level lifecycle readiness flag.

    Main application integration may set:
        app.state.health_lifecycle_ready = False
    before service initialization and:
        app.state.health_lifecycle_ready = True
    after all required services have been initialized successfully.

    If the flag is absent, preserve the historical behavior of this module:
    readiness is derived entirely from configured service objects.
    """
    value = getattr(request.app.state, "health_lifecycle_ready", None)

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    LOGGER.warning(
        "Invalid health_lifecycle_ready state; treating application as not ready."
    )
    return False


def _service_ready(value: Any) -> bool:
    """
    Perform a cheap readiness check on an already-created service.

    Priority:
      1. Explicit `is_ready` attribute/property.
      2. Explicit `ready` attribute/property.
      3. Object existence.

    Callable readiness hooks are intentionally NOT invoked. A health endpoint
    must not accidentally trigger expensive initialization or I/O.
    """
    if value is None:
        return False

    for attribute in ("is_ready", "ready"):
        try:
            marker = getattr(value, attribute, None)
        except Exception:
            return False

        if isinstance(marker, bool):
            return marker

        # Some services expose a cheap boolean-valued property-like callable.
        # Do not invoke callables because doing so could perform work.
        if marker is not None:
            return True

    return True


def _check_services(
    request: Request,
) -> tuple[dict[str, str], bool]:
    """
    Inspect configured application services without executing them.
    """
    app_state = request.app.state
    requirements = _required_services(request)

    checks: dict[str, str] = {}
    all_required_ready = True

    lifecycle_ready = _lifecycle_ready(request)

    # An explicit lifecycle gate is authoritative. This prevents a stale,
    # partially initialized, or placeholder service object from making the
    # application appear ready before main.py has completed startup.
    if lifecycle_ready is False:
        for service_name in requirements:
            checks[service_name] = "unavailable"
        return checks, False

    for service_name, required in requirements.items():
        try:
            value = getattr(app_state, service_name, None)
            ready = _service_ready(value)
        except Exception:
            # Server-side logs may contain the actual exception, while the
            # public response contains only a controlled state.
            LOGGER.exception(
                "Readiness inspection failed for component=%s",
                service_name,
            )
            ready = False

        state = "ready" if ready else "unavailable"
        checks[service_name] = state

        if required and not ready:
            all_required_ready = False

    return checks, all_required_ready


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application liveness",
    description=(
        "Lightweight liveness endpoint. It verifies only that the FastAPI "
        "application can respond; it performs no AI inference or dependency I/O."
    ),
)
async def health(request: Request) -> HealthResponse:
    """
    Liveness probe.

    This endpoint intentionally does not inspect models, indexes, LLMs,
    external services, files, or databases.
    """
    return HealthResponse(
        status="healthy",
        service=_safe_service_name(request),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Application readiness",
    description=(
        "Cheap readiness probe based on already-initialized application "
        "services. Required services unavailable => HTTP 503."
    ),
    responses={
        503: {
            "description": (
                "The API process is alive but one or more required "
                "application services are not ready."
            )
        }
    },
)
async def readiness(request: Request) -> JSONResponse:
    """
    Readiness probe.

    Only application-state inspection is performed. No search, model
    inference, LLM generation, PDF processing, external request, or filesystem
    scan is performed.
    """
    started = time.perf_counter()

    try:
        checks, ready = _check_services(request)
    except Exception:
        # Defensive boundary: health endpoints must remain resilient.
        LOGGER.exception("Unexpected readiness-check failure")
        checks = {}
        ready = False

    payload = ReadinessResponse(
        status="ready" if ready else "not_ready",
        service=_safe_service_name(request),
        checks=checks,
        timestamp=datetime.now(timezone.utc),
    )

    # Successful readiness calls are intentionally not logged individually to
    # avoid monitoring/load-balancer log flooding.
    if not ready:
        LOGGER.warning(
            "Application is not ready; readiness check completed in %.4fs",
            time.perf_counter() - started,
        )

    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if ready
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content=payload.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """
    Model-free FastAPI contract tests.

    No SPECTER2, FAISS, LLM, PDF parser, embedding encoder, network service,
    or project service implementation is loaded.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.state.service_name = DEFAULT_SERVICE_NAME
    app.state.health_lifecycle_ready = True
    app.state.health_required_services = {
        "research_service": True,
        "paper_indexing_service": True,
        "rag_pipeline": True,
    }

    class ReadyService:
        is_ready = True

    app.state.research_service = ReadyService()
    app.state.paper_indexing_service = ReadyService()
    app.state.rag_pipeline = ReadyService()
    app.include_router(router)

    client = TestClient(app)

    # 1. Liveness.
    live = client.get("/health")
    assert live.status_code == 200, live.text
    live_payload = live.json()
    assert live_payload["status"] == "healthy"
    assert live_payload["service"] == DEFAULT_SERVICE_NAME

    # 2. Readiness when all required dependencies are ready.
    ready = client.get("/health/ready")
    assert ready.status_code == 200, ready.text
    ready_payload = ready.json()
    assert ready_payload["status"] == "ready"
    assert ready_payload["checks"]["research_service"] == "ready"
    assert ready_payload["checks"]["paper_indexing_service"] == "ready"
    assert ready_payload["checks"]["rag_pipeline"] == "ready"
    assert ready_payload["timestamp"].endswith("Z")

    # 3. Required dependency failure -> 503.
    app.state.rag_pipeline = None
    not_ready = client.get("/health/ready")
    assert not_ready.status_code == 503, not_ready.text
    not_ready_payload = not_ready.json()
    assert not_ready_payload["status"] == "not_ready"
    assert not_ready_payload["checks"]["rag_pipeline"] == "unavailable"

    # 4. Liveness remains healthy when readiness fails.
    live_while_not_ready = client.get("/health")
    assert live_while_not_ready.status_code == 200
    assert live_while_not_ready.json()["status"] == "healthy"

    # 5. Explicit lifecycle gate overrides otherwise-ready service objects.
    app.state.rag_pipeline = ReadyService()
    app.state.health_lifecycle_ready = False
    lifecycle_not_ready = client.get("/health/ready")
    assert lifecycle_not_ready.status_code == 503, lifecycle_not_ready.text
    assert lifecycle_not_ready.json()["status"] == "not_ready"

    # Restore lifecycle readiness before testing optional dependencies.
    app.state.health_lifecycle_ready = True

    # 6. Optional dependency does not fail readiness.
    app.state.health_required_services = {
        "research_service": True,
        "paper_indexing_service": True,
        "rag_pipeline": True,
        "llm_client": False,
    }
    optional_missing = client.get("/health/ready")
    assert optional_missing.status_code == 200
    assert optional_missing.json()["status"] == "ready"
    assert optional_missing.json()["checks"]["llm_client"] == "unavailable"

    # 7. OpenAPI route generation.
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert "/health" in paths
    assert "/health/ready" in paths

    # 8. No obvious secret/path fields in the public payload.
    serialized = str(optional_missing.json()).lower()
    forbidden_markers = (
        "api_key",
        "password",
        "secret",
        "token",
        "c:\\",
        "/users/",
    )
    assert not any(marker in serialized for marker in forbidden_markers)

    print("backend/api/health.py self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()