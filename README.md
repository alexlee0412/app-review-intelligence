# App Review Intelligence

An evidence-grounded app review analytics platform combining semantic search, SQL filtering, and trend aggregation to surface emerging customer complaints.

## Overview

The planned web application will help product, research, and insight teams:

- Search app reviews by meaning
- Filter reviews by structured metadata
- Measure complaint trends across apps
- Inspect representative raw review evidence
- Trace how each analysis result was generated

## Example Analysis

"Find subscription cancellation-related complaints from US 1-2 star reviews during the most recent three complete months, calculate monthly review counts and month-over-month growth by app, and return representative reviews."

## Planned Capabilities

- Semantic review search
- SQL metadata filtering
- Monthly trend aggregation
- Month-over-month growth calculation
- Representative review evidence
- Transparent analysis trace
- Interactive analytical dashboard

## Planned Stack

- React and TypeScript
- FastAPI and Python
- PostgreSQL and pgvector
- SQLAlchemy and Pydantic
- Docker Compose

## Project Status

**Status: Initial backend scaffold**

The repository contains the product and technical plan, plus an initial FastAPI backend and local PostgreSQL/pgvector development environment (configuration, database session management, and health endpoints only). Review analysis, semantic search, embeddings, ingestion, and the frontend are not implemented yet.

## Development

Requires Docker and Docker Compose for the full local stack, or Python 3.12 to run the backend directly.

Copy the example environment file:

```bash
cp backend/.env.example backend/.env
```

Start the local stack (PostgreSQL with pgvector, and the API):

```bash
docker compose up --build
```

Open the interactive API documentation:

```
http://localhost:8000/docs
```

Check application health (no database dependency):

```bash
curl http://localhost:8000/health
```

Check database health:

```bash
curl http://localhost:8000/health/db
```

Stop the stack:

```bash
docker compose down
```

To work on the backend outside Docker, install it in a virtual environment (Python 3.12):

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the backend test suite:

```bash
cd backend
pytest
```

## Documentation

[View the complete project plan](./PROJECT_PLAN.md)

## Data and Privacy

This project will use public, synthetic, or portfolio-safe sample data. It will not include confidential company data.
