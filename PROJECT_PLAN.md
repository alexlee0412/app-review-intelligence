# Project Plan: App Review Intelligence

This document is the source of truth for the initial product and technical plan for App Review Intelligence. The repository is currently in the planning stage; no application code has been implemented yet.

## Project Title

App Review Intelligence

## Project Overview

App Review Intelligence is an evidence-grounded analytics system for mobile app reviews.

It combines:

- Semantic vector search over review text
- Relational SQL filtering
- SQL-based aggregation and trend analysis
- Representative raw reviews as supporting evidence
- Transparent filters and query traceability

The project is intended to support app review analysis, Voice of Customer analysis, UXR, product insights, and AI-assisted research workflows.

The product should demonstrate more than a basic RAG chatbot. Quantitative calculations must be performed through deterministic application logic and SQL rather than generated from free-form LLM reasoning.

## Problem Statement

Product and research teams may receive thousands of app reviews across different apps, countries, ratings, versions, and time periods.

Manually identifying whether a specific complaint is increasing is slow and difficult because:

- Users describe the same issue using different language
- Relevant reviews must be filtered by structured metadata
- Trends require accurate counts and period-over-period calculations
- Analytical conclusions must remain traceable to real review evidence

App Review Intelligence addresses this by combining semantic retrieval with structured filtering and deterministic aggregation.

## Primary MVP Use Case

The first complete vertical slice should answer this analytical request:

> "Find subscription cancellation-related complaints from US 1-2 star reviews during the most recent three complete months, calculate monthly review counts and month-over-month growth by app, and return representative reviews."

The initial implementation may use fixed analysis parameters before natural-language query interpretation is introduced.

## Expected Analysis Output

The system should eventually return:

1. Executive summary
2. Total number of semantically matched reviews
3. Monthly complaint counts by app
4. Month-over-month growth by app
5. Key complaint themes
6. Three representative reviews per app
7. Review evidence metadata:
   - App name
   - App version
   - Rating
   - Country
   - Created date
   - Semantic similarity score
8. Applied filters
9. SQL template or executed-query trace

Example conclusions may include:

- App A: subscription cancellation complaints increased by 38%
- App B: subscription cancellation complaints increased by 17%

Example complaint themes may include:

- Cancellation options are difficult to find
- Unexpected billing after a free trial
- Refund procedures are unclear

## Core System Behavior

Planned request flow:

```
User analytical request
  -> validated analysis parameters
  -> query embedding generation
  -> SQL metadata filters
  -> vector similarity search
  -> SQL aggregation
  -> growth calculation
  -> representative review selection
  -> deterministic response formatting
  -> optional LLM-generated narrative summary
```

Responsibility of each layer:

- Vector search finds semantically related reviews.
- SQL filters app, country, rating, version, and date metadata.
- SQL and application logic calculate counts and trends.
- The LLM may interpret natural-language requests and summarize verified results.
- The LLM must not directly execute arbitrary SQL.
- Raw reviews and metadata provide evidence for conclusions.

## Planned Technology Stack

**Backend:**

- Python 3.12
- FastAPI
- SQLAlchemy 2.x
- Pydantic
- pydantic-settings
- PostgreSQL
- pgvector
- pandas

**Frontend:**

- React
- TypeScript
- Vite
- Recharts

**Infrastructure:**

- Docker Compose for local PostgreSQL and pgvector
- Environment-based configuration
- Replaceable embedding provider interface

**Embedding configuration for the initial plan:**

- Provider: OpenAI
- Model: text-embedding-3-small
- Default dimension: 1536
- Similarity metric: cosine distance

The embedding dimension must remain consistent across:

- The selected embedding model
- Application configuration
- SQLAlchemy vector column
- PostgreSQL schema
- Stored embedding validation

## Data Strategy

The initial MVP should use a reproducible static dataset rather than real-time review collection.

Target dataset:

- Approximately 5,000-20,000 app reviews
- Approximately 3-5 mobile apps
- App metadata
- Review version, rating, country, date, title, and body
- Optional public or synthetic release notes for later extensions

The initial dataset must not contain confidential company data, internal SNOW data, personal data, or proprietary code.

## Planned Data Model

### apps

- app_id
- app_name
- category
- platform

### reviews

- review_id
- app_id
- version
- rating
- country
- created_at
- title
- body
- embedding
- source

The embedding column should use a fixed dimension: `VECTOR(1536)`.

### query_runs

- query_run_id
- user_query
- parsed_intent
- applied_filters
- generated_sql or sql_template
- result_summary
- created_at

`query_runs` provides analysis traceability and stores query execution metadata separately from review retrieval.

## Growth-Rate Behavior

Explicit handling for a previous-period count of zero.

When:

- `previous_month_count = 0`
- `current_month_count > 0`

Return:

- `growth_rate = null`
- `growth_status = "new"`

Do not return infinity or an artificial percentage.

When both current and previous counts are zero, return an appropriate unchanged or no-data status.

Growth-rate unit tests should cover:

- Normal positive growth
- Negative growth
- No change
- Previous count equal to zero
- Both periods equal to zero

## Representative Review Selection

The system should initially select three representative reviews per app.

Selection should consider:

- Semantic relevance
- Recency
- Review metadata
- Duplicate or near-duplicate removal
- Evidence diversity

The MVP may begin with deterministic similarity ranking and text deduplication before adding more advanced diversification.

## Vector Indexing Decision

Do not require an approximate vector index during the initial implementation.

For the initial dataset:

- Store embeddings as `VECTOR(1536)`
- Begin with exact cosine-distance search
- Validate retrieval correctness before optimizing performance
- Add an approximate index only if measurements show a meaningful need

If an approximate index is introduced later, it should be documented as a separate optimization decision rather than included as a required initial feature.

Possible future option:

- HNSW with `vector_cosine_ops`

An IVFFlat index is not a mandatory initial schema requirement.

## Planned Repository Structure

```
app-review-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── db.py
│   │   ├── models/
│   │   │   ├── app.py
│   │   │   ├── review.py
│   │   │   └── query_run.py
│   │   ├── schemas/
│   │   │   └── review.py
│   │   ├── api/
│   │   │   ├── reviews.py
│   │   │   └── apps.py
│   │   ├── repositories/
│   │   │   ├── review_repository.py
│   │   │   └── query_run_repository.py
│   │   └── services/
│   │       ├── embedding_service.py
│   │       ├── review_search.py
│   │       ├── aggregation.py
│   │       └── answer_generator.py
│   ├── scripts/
│   │   ├── load_reviews.py
│   │   └── generate_embeddings.py
│   ├── tests/
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── api/
│       └── types/
├── data/
│   └── sample_reviews.csv
├── database/
│   └── schema.sql
├── docker-compose.yml
├── PROJECT_PLAN.md
└── README.md
```

Responsibilities of the main directories:

- `core`: configuration, database engine, sessions, and dependencies
- `models`: SQLAlchemy ORM models
- `schemas`: Pydantic request and response contracts
- `api`: FastAPI route definitions
- `repositories`: database access and persistence
- `services`: search, aggregation, embedding, and response logic
- `scripts`: repeatable ingestion and embedding jobs
- `tests`: deterministic business-logic and API tests

Natural-language intent parsing is not included in the initial repository structure. A future `intent_parser.py` may be added when natural-language query support is implemented.

## Initial API Boundary

Planned initial endpoint:

```
POST /api/v1/reviews/analyze
```

The first version may use fixed parameters:

- Country: US
- Ratings: 1 and 2
- Time range: most recent three complete months
- Semantic topic: subscription cancellation

The response contract should contain structured JSON rather than only free-form prose.

Suggested top-level response fields:

- `analysis`
- `matched_review_count`
- `app_trends`
- `complaint_themes`
- `evidence`
- `applied_filters`
- `query_trace`

## CSV Ingestion Requirements

The ingestion process should eventually:

- Validate required columns
- Normalize countries
- Validate ratings
- Parse dates consistently
- Normalize empty title and body values
- Insert apps and reviews
- Support safe repeat execution
- Prevent duplicate review records
- Record invalid rows separately
- Avoid embedding generation during basic ingestion

## Embedding Generation Requirements

The embedding process should eventually:

- Combine review title and body
- Process reviews in batches
- Skip rows that already contain embeddings
- Validate returned vector dimensions
- Store embeddings in PostgreSQL
- Use an embedding-provider interface
- Support provider replacement without rewriting retrieval logic
- Avoid committing API keys or generated secrets

## Testing Priorities

Planned tests for:

- Date-range calculation
- Growth-rate calculation
- Previous-count-zero behavior
- CSV validation
- Request schemas
- Response schemas
- Representative review deduplication
- Repository query behavior
- API success and validation responses

## MVP Scope

- Static review data ingestion
- Review embedding generation
- Semantic retrieval
- Structured SQL filters
- Monthly aggregation
- Month-over-month growth
- Representative evidence reviews
- Query traceability
- Structured API response
- Analytical frontend
- Natural-language parsing only after the deterministic pipeline is reliable

## Explicit Non-Goals

Excluded from the initial MVP:

- Authentication
- Multi-user accounts
- Real-time App Store collection
- Production-scale crawling
- LangChain
- General-purpose agent frameworks
- Arbitrary LLM-generated SQL execution
- Autonomous actions
- Release-note correlation
- KPI correlation
- Ranking correlation
- Multi-country production support
- Billing or subscription systems
- Enterprise deployment infrastructure

## Future Extensions

Future possibilities, not MVP commitments:

- Release-note semantic matching
- Complaint changes by app version
- KPI and ranking correlation
- Multi-country analysis
- Saved analyses
- Scheduled reports
- Review-theme clustering
- Product opportunity recommendations
- Hybrid keyword and vector retrieval
- Retrieval reranking
- Additional embedding providers

## Success Criteria

The initial system should be considered successful when it can:

- Reliably identify reviews related to a fixed complaint topic
- Apply accurate structured metadata filters
- Produce reproducible monthly counts
- Calculate growth with documented edge-case behavior
- Return representative real review evidence
- Expose applied filters and query trace
- Separate retrieval, aggregation, persistence, and formatting responsibilities
- Produce the same quantitative result without relying on free-form LLM reasoning
