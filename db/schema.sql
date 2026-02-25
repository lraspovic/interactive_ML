-- =============================================================================
-- Interactive ML – Land Cover Mapping
-- Full database schema (auto-generated from Alembic migrations 0001–0003)
--
-- Usage (manual recreation):
--   psql postgresql://lcmap:lcmap@localhost:5433/lcmap -f db/schema.sql
--
-- NOTE: When running via docker compose the backend entrypoint runs
--       "alembic upgrade head" automatically, so this file is only needed
--       for manual / out-of-Docker database setup.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- Alembic version tracking (must exist before migrations mark themselves done)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- ---------------------------------------------------------------------------
-- projects
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    id              UUID                     NOT NULL DEFAULT gen_random_uuid(),
    name            VARCHAR(255)             NOT NULL,
    description     TEXT,
    task_type       VARCHAR(64)              NOT NULL DEFAULT 'classification',
    aoi_geometry    geometry(POLYGON, 4326),
    imagery_url     TEXT,
    available_bands JSONB,
    enabled_indices JSONB,
    resolution_m    INTEGER                  NOT NULL DEFAULT 10,
    model_config    JSONB,
    created_at      TIMESTAMPTZ              NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ              NOT NULL DEFAULT now(),
    CONSTRAINT projects_pkey        PRIMARY KEY (id),
    CONSTRAINT projects_name_key    UNIQUE      (name)
);

CREATE INDEX IF NOT EXISTS idx_projects_name ON projects (name);

-- ---------------------------------------------------------------------------
-- classes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS classes (
    id            SERIAL       NOT NULL,
    project_id    UUID         NOT NULL,
    name          VARCHAR(128) NOT NULL,
    color         VARCHAR(7)   NOT NULL,   -- hex, e.g. '#2d6a4f'
    display_order INTEGER      NOT NULL DEFAULT 0,
    CONSTRAINT classes_pkey       PRIMARY KEY (id),
    CONSTRAINT classes_project_fk FOREIGN KEY (project_id)
        REFERENCES projects (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_classes_project_id ON classes (project_id);

-- ---------------------------------------------------------------------------
-- training_samples
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_samples (
    id         UUID                    NOT NULL DEFAULT gen_random_uuid(),
    project_id UUID,
    geometry   geometry(POLYGON, 4326) NOT NULL,
    label      VARCHAR(128)            NOT NULL,
    class_id   INTEGER,
    image_ref  TEXT,
    metadata   JSONB,
    created_at TIMESTAMPTZ             NOT NULL DEFAULT now(),
    CONSTRAINT training_samples_pkey       PRIMARY KEY (id),
    CONSTRAINT training_samples_project_fk FOREIGN KEY (project_id)
        REFERENCES projects (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_training_samples_label      ON training_samples (label);
CREATE INDEX IF NOT EXISTS idx_training_samples_project_id ON training_samples (project_id);

-- ---------------------------------------------------------------------------
-- training_features
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_features (
    id                 UUID         NOT NULL DEFAULT gen_random_uuid(),
    training_sample_id UUID         NOT NULL,
    project_id         UUID         NOT NULL,
    item_id            TEXT         NOT NULL,   -- STAC item identifier
    collection         VARCHAR(128) NOT NULL,
    feature_names      JSONB        NOT NULL,   -- ordered list of band/index names
    n_pixels           INTEGER      NOT NULL,
    feature_data       BYTEA        NOT NULL,   -- numpy.save() serialised float32 array
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT training_features_pkey     PRIMARY KEY (id),
    CONSTRAINT training_features_sample_fk FOREIGN KEY (training_sample_id)
        REFERENCES training_samples (id) ON DELETE CASCADE,
    CONSTRAINT training_features_project_fk FOREIGN KEY (project_id)
        REFERENCES projects (id) ON DELETE CASCADE
);

-- Fast lookup: all features for a project + scene
CREATE INDEX IF NOT EXISTS idx_training_features_project_scene
    ON training_features (project_id, item_id);

-- Enforce one row per sample per scene (upsert guard)
CREATE UNIQUE INDEX IF NOT EXISTS idx_training_features_sample_scene
    ON training_features (training_sample_id, item_id, collection);

-- ---------------------------------------------------------------------------
-- Stamp all migrations as applied (so Alembic doesn't re-run them)
-- ---------------------------------------------------------------------------
INSERT INTO alembic_version (version_num)
VALUES ('0003')
ON CONFLICT DO NOTHING;
