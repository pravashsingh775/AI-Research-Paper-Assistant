# Setup

Requirements: Python 3.12+, Node.js 20+, and network access for OpenAlex. Copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY` for model-backed analysis.

```powershell
python -m pip install -e ".[dev]"
npm install --prefix apps/web
python -m uvicorn apps.api.app.main:app --reload --port 8000
npm run dev:web
```

Open `http://localhost:3000`. For a stable compiled run, stop the dev server, run `npm run build --prefix apps/web`, then `npm run start --prefix apps/web`.

`SEMANTIC_SCHOLAR_API_KEY` is optional. OpenAlex is used without a key. Never put provider keys in frontend environment variables.
