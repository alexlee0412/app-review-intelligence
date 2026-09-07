# App Review Intelligence frontend

A local, single-page Next.js App Router client. Requires Node.js 24 and npm.

From this directory:

```sh
npm install
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` to the backend origin. Development
defaults are documented in `.env.example`. This value is public and must contain no
credentials. Restart the development server after changing it; for a production
build, set it before `npm run build`. Next.js prints the frontend URL at startup.
To change the development port, set `PORT` in your shell before running Next.js
(Next does not read the listening port from `.env.local`).

The separately maintained backend must expose `POST /api/v1/reviews/ask`, accept
`{ "question": "..." }`, and allow the frontend origin through CORS when served from
a different origin. No backend, proxy route, or sample response server is included.
Requests have no client deadline and can take tens of seconds. The in-flight guard
prevents repeated submissions until the request completes.

`lib/answer.ts` validates the complete AnswerResponse shape before rendering. Zod is
the only runtime dependency beyond Next.js and React; it also supplies the TypeScript
contract. Empty metrics and nullable averages/dates are supported. Metrics use the
backend values directly. The overall average is formatted to two decimal places;
no factual values are calculated in the browser. Evidence excerpts and narrative
line breaks are preserved. Limitations and warnings are always expanded. Trace and
per-app metrics are available in disclosure controls.

The client discards HTTP error bodies and maps status codes to friendly copy. Invalid
JSON or an invalid response shape follows the same retry path. Unknown response
fields are stripped by validation. Test data in `tests/fixtures.ts` is entirely
synthetic; tests mock only the browser's fetch boundary and require no backend,
database, network, or provider credentials.

```sh
npm run lint
npm run type-check
npm test
npm run build
```

Linting is an explicit check, separate from the production build, as documented in
the [Next.js installation guide](https://nextjs.org/docs/app/getting-started/installation).
The test suite covers rendering, runtime validation, errors, retry, and request
deduplication. Generated files, local environment files, dependencies, and build
output are ignored within this directory.
