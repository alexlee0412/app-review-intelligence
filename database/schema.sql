-- Database bootstrap for local development.
-- This file must stay in parity with the SQLAlchemy models in backend/app/models.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS apps (
    app_id     VARCHAR(64)  PRIMARY KEY,
    -- Not unique: the same product may exist across platforms or stores.
    app_name   VARCHAR(255) NOT NULL,
    category   VARCHAR(128),
    platform   VARCHAR(32)  NOT NULL DEFAULT 'ios',
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_apps_app_name ON apps (app_name);

CREATE TABLE IF NOT EXISTS reviews (
    review_id   VARCHAR(64) PRIMARY KEY,
    app_id      VARCHAR(64) NOT NULL,
    version     VARCHAR(32),
    rating      INTEGER     NOT NULL,
    -- ISO 3166-1 alpha-2 store country code.
    country     VARCHAR(2)  NOT NULL,
    -- When the user originally posted the review.
    created_at  TIMESTAMPTZ NOT NULL,
    title       TEXT,
    body        TEXT        NOT NULL,
    -- Nullable: reviews are ingested before embeddings are generated.
    embedding   VECTOR(1536),
    source      VARCHAR(64),
    -- When the review entered App Review Intelligence.
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fk_reviews_app_id_apps
        FOREIGN KEY (app_id) REFERENCES apps (app_id) ON DELETE RESTRICT,
    CONSTRAINT ck_reviews_rating_range CHECK (rating BETWEEN 1 AND 5)
);

-- Primary MVP filter: country + rating + date range.
CREATE INDEX IF NOT EXISTS ix_reviews_country_rating_created_at
    ON reviews (country, rating, created_at);
CREATE INDEX IF NOT EXISTS ix_reviews_app_id ON reviews (app_id);
-- No index on version: per-version analysis is a future extension, and no
-- approximate vector index on embedding: retrieval starts as exact cosine
-- search and will be measured before being optimized.

CREATE TABLE IF NOT EXISTS query_runs (
    -- UUIDs are generated application-side, so no server default is declared.
    query_run_id    UUID        PRIMARY KEY,
    -- Nullable: the first analysis flow uses fixed parameters rather than a
    -- natural-language request.
    user_query      TEXT,
    parsed_intent   JSONB,
    applied_filters JSONB,
    sql_template    TEXT,
    result_summary  JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
