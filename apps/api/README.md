# API service

FastAPI service entry point and route implementation are currently in `app/main.py`; request models are defined in the same module. Supporting logic is in `app/services/`, `app/core/`, and `app/providers/`.

The API currently serves `/api/...` routes rather than a versioned `/api/v1` surface. See [`../../docs/API.md`](../../docs/API.md) for the implemented routes and [`../../docs/PROJECT_REVIEW.md`](../../docs/PROJECT_REVIEW.md) for known gaps. Keep new route logic small and extract use cases into `app/services/` when behavior is covered by tests.
