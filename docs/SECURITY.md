# Security and Privacy Notes

## Existing controls

- Passwords use `pwdlib` hashing; access tokens are signed JWTs.
- Private paper, collection, job, and conversation routes use authenticated user dependencies and ownership checks.
- PDF validation checks file size and PDF structure/signature before processing; extraction is performed by the worker.
- Uploaded PDFs are stored outside relational rows using local or S3-compatible storage.
- API request correlation IDs are propagated in response headers.
- Analytical endpoints now require authentication. Readiness failures are logged server-side and return only boolean dependency status to clients.

These controls have not been independently penetration-tested. Integration tests for cross-user access require the live stack.

## Configuration and deployment risks

- Development defaults exist for JWT, database, Redis, and object-storage configuration. Compose includes development credentials. Do not expose this Compose configuration as a production deployment.
- The production JWT guard is applied when `APP_ENV` is set to `production` or `prod`. Set this explicitly and supply a strong secret through a secret manager.
- The Compose file uses floating container tags for some dependencies and does not provide production backup, TLS, network isolation, or secret-management configuration.
- Python dependencies are version ranges except for selected ML packages; there is no Python lockfile.
- API rate limits and request-cost controls are not implemented.

## Client/provider key handling

The web app stores user-supplied LLM keys in browser `localStorage` and sends them to the API in request headers. This exposes keys to any successful same-origin script injection and is not suitable for high-assurance deployments. Prefer server-side secret storage or a short-lived, explicitly consented provider session. Do not log request headers or provider credentials.

## Document processing and prompt injection

PDF content is untrusted. Keep prompt instructions separate from retrieved content, validate citations against retrieved evidence, limit document size/pages and model context, and isolate parsers from credentials and unrelated files. The current implementation has size/page validation and prompt boundaries, but does not run the parser in a hardened sandbox and does not provide OCR.

## Operational requirements before deployment

1. Generate unique secrets and disable all development credentials.
2. Restrict CORS, network access, database roles, and object-store permissions.
3. Configure TLS, backups, restore drills, retention, and deletion verification.
4. Add rate limiting, upload resource limits, and job retry/dead-letter monitoring.
5. Review logs and readiness responses for sensitive data.
6. Run the integration and authorization suites against the exact release image.
