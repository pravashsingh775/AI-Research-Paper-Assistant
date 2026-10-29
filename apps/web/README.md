# Web application

Next.js 15 / React 19 interface for the research workspace. Route experiences are in `app/`, shared UI is in `components/`, and the typed API client is in `lib/api.ts`.

Run `npm ci --prefix apps/web`, then `npm run dev:web` from the repository root. Set `NEXT_PUBLIC_API_URL` to the API origin when it is not `http://localhost:8000`. The production build command is `npm run build`.

The web app currently has no non-interactive ESLint configuration. `npm --prefix apps/web run lint` opens the Next.js ESLint setup prompt.
