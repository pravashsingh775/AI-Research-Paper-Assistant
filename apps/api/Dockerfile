FROM python:3.12-slim
WORKDIR /app
ENV TMPDIR=/var/tmp PIP_CACHE_DIR=/var/cache/pip
COPY pyproject.toml .
RUN mkdir -p /var/tmp /var/cache/pip && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple .
COPY apps ./apps
COPY infra ./infra
COPY alembic.ini .
CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000"]
