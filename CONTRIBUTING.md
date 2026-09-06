# Contributing to Lumen Research

Thank you for your interest in contributing to **Lumen Research**! We welcome contributions from software engineers, ML researchers, and academic tool builders.

---

## 🛠️ Development Setup

### Prerequisites
- **Docker Desktop** (Docker Engine + Compose v2)
- **Python 3.11+**
- **Node.js 20+** (npm 10+)
- **Git**

### 1. Clone & Environment Configuration
```bash
git clone https://github.com/pravashsingh775/AI-Research-Paper-Assistant.git
cd AI-Research-Paper-Assistant
cp .env.example .env
```

### 2. Start Full Infrastructure Stack
```bash
# Starts PostgreSQL (pgvector), Redis, MinIO, Backend, Worker, Frontend
docker compose -f infra/docker/compose.yml up -d --build --wait
```

### 3. Verify Health & Readiness
```bash
# Fast liveness probe
curl http://localhost:8000/liveness

# Deep multi-tier readiness probe (Postgres, Redis, MinIO)
curl http://localhost:8000/readiness
```

---

## 🧪 Testing & Verification

Before submitting any Pull Request, ensure all tests pass:

```bash
# 1. Run full backend test suite
python -m pytest apps/api/tests/ -v

# 2. Run text normalization & LaTeX regression suite
python -m pytest apps/api/tests/test_text_normalization_regression.py -v

# 3. Run quantitative RAG evaluation benchmark
python scripts/evaluate_rag.py

# 4. Verify Next.js frontend production build
npm --prefix apps/web run build

# 5. Run end-to-end multi-feature audit
python scripts/test_all_features.py
```

---

## 📐 Architecture & Coding Guidelines

1. **Evidence-Grounded AI Principle:** 
   - Never generate ungrounded scientific claims. Responses must cite chunk, page, and section. If evidence is insufficient, the system must faithfully abstain.
2. **Text Normalization Safety:**
   - Any raw text extracted from academic PDFs must pass through `apps/api/app/services/text_normalization.py` to eliminate NUL bytes (`0x00`) while strictly preserving scientific symbols (`∑`, `√`, `|ψ⟩`, etc.).
3. **Database Integrity:**
   - Always create new Alembic migrations for schema changes:
     ```bash
     alembic revision -m "description_of_change"
     ```
4. **Worker Isolation:**
   - Long-running or heavy PDF processing must be handled asynchronously via the Redis worker queue (`research-paper-jobs`), using isolated transaction sessions for failure states.

---

## 🌿 Git Branch & PR Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes following Conventional Commits:
   ```bash
   git commit -m "feat(rag): add multi-chunk cross-encoder reranking"
   ```
3. Push to your branch:
   ```bash
   git push origin feat/your-feature-name
   ```
4. Open a Pull Request against `main`. Ensure all automated CI checks pass.

---

## 📜 License
By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

