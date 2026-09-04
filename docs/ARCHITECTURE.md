# Architecture

How App Review Intelligence is put together: system boundaries, data flow, the contracts
between components, and the decisions behind them.

This describes the design as it currently stands, not its history. Keep it current when a
decision below changes, and keep it free of secrets, machine-specific paths, and commit
hashes.

## Product goal

An evidence-grounded analytics system for mobile app reviews: semantic retrieval over
review text, combined with relational metadata filtering and deterministic aggregation, so
every conclusion traces back to real review evidence. Quantitative results come from SQL
and application logic, never from free-form model reasoning.

## Deployment target

The MVP is deployed on **Vercel**. Docker Compose is local development and integration
infrastructure only — it is not the production architecture.

## System boundaries

```
Apify dump or dataset
  -> source adapter (local file | REST)      operator-run, not deployed
  -> normalization (pure, tested)            operator-run, not deployed
  -> idempotent upsert -> PostgreSQL         operator-run, not deployed
  -> embedding backfill                      operator-run, not deployed
  ------------------------------------------ deployment boundary
  -> exact cosine kNN + SQL metadata filters deployed, request-driven
  -> evidence payload + query trace          deployed, request-driven
```

Only the read path is deployed. Ingestion is a batch job the operator runs against
whichever database `APP_DATABASE_URL` points at — the same code seeds local and
production.

## Integration boundary

**The PostgreSQL review row is the integration boundary between ingestion and retrieval.**
Ingestion never calls retrieval; retrieval never calls ingestion. They meet only at
persisted rows. This is what allows the two roles to be implemented independently.

## Data flow and storage assumptions

- Core tables: `apps`, `reviews`, `query_runs`.
- `reviews.review_id` is the primary key and is derived deterministically from the source
  record, which is what makes re-ingestion idempotent. No surrogate key, no extra dedupe
  column.
- `reviews.embedding` is `VECTOR(1536)` and **nullable** — rows are ingested before they
  are embedded. The backfill job fills exactly the null rows.
- Re-ingesting unchanged text preserves existing embeddings. Re-ingesting changed title or
  body invalidates the stale embedding by setting it to null.
- Retrieval uses **exact cosine distance**. No approximate vector index: the MVP dataset is
  roughly 5k-20k rows, and indexing is deferred until measurement justifies it.
- `query_runs` stores the trace for each executed analysis. It never stores embedding
  vectors, credentials, or connection strings.

## Shared contracts

- **The database schema and ORM models are treated as frozen** and changed deliberately.
  They are the contract between every component.
- **Embedding provider interface** — a provider exposes a name, a dimension, a
  production-grade flag, batch embedding, and single-query embedding. Retrieval consumes a
  callable plus that metadata; it does not import provider implementations at module scope.
  Swapping providers must not require changes to retrieval architecture.
- **Embedding dimension must agree across four places**: application configuration, the ORM
  model constant, the SQL schema column, and what the provider actually returns.
- **Search API** — `POST /api/v1/reviews/search` returns evidence rows with similarity
  scores, the applied filters, a query trace, and a warnings list.
- Date ranges are **half-open** `[from, to)` everywhere.

## Embedding providers

- A deterministic **fake provider is the development and test default**. Its vectors are
  hash-seeded and carry **no semantic meaning**. It exists so tests are deterministic,
  local integration is runnable without a paid key, and the architecture is independently
  verifiable.
- **Real semantic retrieval in production requires a production-grade embedding provider.**
  The fake provider must never serve production traffic: configuration fails loudly rather
  than degrading silently, and any response served by a non-production provider carries an
  explicit warning.
- **The reviews table holds vectors from exactly one provider and model at a time.** Mixing
  providers makes cosine distance meaningless. Switching providers requires re-embedding
  every row.

## Production vs local development

| | Local development | Production (Vercel) |
|---|---|---|
| Database | Docker Compose PostgreSQL + pgvector | managed PostgreSQL with pgvector, pooled connection string |
| API | uvicorn, optionally in Compose | serverless function, short-lived and stateless |
| Ingestion | operator runs the scripts | operator runs the same scripts against the managed database |
| Embeddings | deterministic fake provider | production-grade provider, required |
| Config | `.env` (gitignored) | Vercel environment variables |

Production requires a **managed PostgreSQL with pgvector** — this is the only external
infrastructure the MVP needs. No Redis, queue, object store, cron, or worker process.

## Deferred work

- Serverless connection pooling for the database engine (a module-level pool per cold start
  exhausts connections).
- Vercel entrypoint and Python runtime dependency manifest.
- Provisioning the managed database, enabling the vector extension, applying the schema.
- Monthly aggregation, month-over-month growth, complaint theme synthesis, and LLM
  narrative summaries — planned, and deliberately outside the current scope.
- Natural-language query parsing, and the React frontend.
- Approximate vector indexing, only if measurement justifies it.

## Design decisions and rationale

- The database row is the integration boundary; ingestion and retrieval never call each
  other directly.
- Schema changes are the expensive thing. Prefer a design that needs none.
- Growth rate with a zero previous period returns null with a "new" status — never
  infinity and never a fabricated percentage.
- The LLM never executes arbitrary SQL and never produces quantitative results.
- Ingestion is deliberately not deployed; making it request-driven or scheduled was
  considered and rejected for the MVP.
- Deterministic ordering in search requires a full tiebreak, not similarity alone.
