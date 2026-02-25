# AGENTS.md

You are an AI coding agent working on an interactive machine learning web application
for remote sensing image classification. The app is **project-based**: each project
defines its own area of interest, imagery source, spectral features, class scheme,
and model — making it applicable to vegetation mapping, urban analysis, crop
monitoring, coastal/wetland assessment, burned-area detection, and more.

The core loop is:
create/load project → draw/edit training polygons on a map → retrain a
multi-class classifier → see live predictions on the map → repeat with active
learning guidance.

Keep the codebase minimal and focused. Do not introduce complexity that isn't
required by the features described below.

---

## Architecture overview

```
docker-compose
├── frontend/     React app with web map (Leaflet)
├── backend/      FastAPI (ASGI, uvicorn)
├── db/           PostgreSQL 15 + PostGIS 3
└── ml/           ML worker (Python); may be a FastAPI background task
                  or a standalone container depending on current design
```

- The frontend sends GeoJSON polygons as training data and class labels to the backend.
- The backend stores them in PostGIS and triggers model retraining.
- After retraining, the backend exposes a prediction endpoint (tile or polygon).
- The frontend renders prediction results and active-learning uncertainty layers
  directly on the map.
- All services communicate via REST/JSON; no message broker unless already present.

---

## Services and ports (defaults)

| Service  | Port | Entry point                 |
| -------- | ---- | --------------------------- |
| frontend | 3000 | `npm run dev`               |
| backend  | 8000 | `uvicorn main:app --reload` |
| db       | 5432 | PostgreSQL + PostGIS        |

If a `docker-compose.yml` exists, always prefer it over running services manually.

---

## Setup

```bash
docker compose up --build          # start everything
docker compose down -v             # stop and remove volumes
```

Backend-only (for quick iteration):

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend-only:

```bash
cd frontend
npm install
npm run dev
```

---

## Testing

Run tests for every non-trivial change before suggesting a commit.

- Backend: `cd backend && pytest`
- Frontend: `cd frontend && npm test`

Test files live next to the code they test or in a `tests/` directory.
Do not remove existing tests. Add tests for new endpoints and ML utility functions.

---

## Database

- Engine: **PostgreSQL + PostGIS** (container `db`).
- Connection string is read from the environment variable `DATABASE_URL`
  (format: `postgresql+asyncpg://user:pass@db:5432/lcmap`).
- All geospatial columns use **EPSG:4326** unless a migration explicitly
  defines another CRS — do not change CRS silently.
- Schema is managed via Alembic migrations in `backend/alembic/`.
  To create a migration: `alembic revision --autogenerate -m "description"`
  To apply: `alembic upgrade head`
- Never issue raw `DROP TABLE`, `TRUNCATE`, or destructive DDL outside migrations.

### Core tables (reference)

- `projects` — project configuration: name, description, AOI geometry, imagery
  source URL, available bands list, enabled spectral indices, task type, model
  config JSON, created/updated timestamps.
- `classes` — per-project class definitions: id, project_id FK, name, color hex,
  display order. Classes are project-scoped; there are no global hardcoded classes.
- `training_samples` — polygon geometry + class label + project_id FK + source
  image reference + metadata.
- `predictions` — output polygons or raster references (tile path / COG path)
  per model version + project_id FK.
- `model_versions` — trained model metadata: version id, project_id FK,
  timestamp, accuracy metrics, model type, hyperparameters JSON, file path.
- `uncertainty_tiles` — per-tile entropy or margin scores for active learning,
  project-scoped.

---

## Backend (FastAPI)

Entry: `backend/main.py`. Routers live in `backend/app/routers/`.

### Minimal endpoint surface

Keep endpoints strictly minimal. The current intended endpoints are:

| Method | Path                             | Purpose                                   |
| ------ | -------------------------------- | ----------------------------------------- |
| GET    | `/health`                        | Liveness check                            |
| POST   | `/projects`                      | Create a new project (wizard submit)      |
| GET    | `/projects`                      | List all projects                         |
| GET    | `/projects/{id}`                 | Get a single project (config + classes)   |
| PATCH  | `/projects/{id}`                 | Update project config                     |
| DELETE | `/projects/{id}`                 | Delete project and all associated data    |
| GET    | `/projects/{id}/classes`         | List classes for a project                |
| GET    | `/imagery/tiles/{z}/{x}/{y}`     | Serve base imagery tiles to the map       |
| POST   | `/training-samples`              | Save new annotated polygons               |
| GET    | `/training-samples`              | List all training samples (GeoJSON)       |
| DELETE | `/training-samples/{id}`         | Remove a sample                           |
| POST   | `/train`                         | Trigger model retraining                  |
| GET    | `/train/status`                  | Polling endpoint for training progress    |
| POST   | `/predict`                       | Run prediction on AOI (returns tile ref)  |
| GET    | `/predictions/tiles/{z}/{x}/{y}` | Serve prediction tiles                    |
| GET    | `/uncertainty/tiles/{z}/{x}/{y}` | Serve uncertainty tiles (active learning) |

Do not add new endpoints unless explicitly requested. Extend existing ones instead.

### Conventions

- Use **Pydantic v2** models for all request/response schemas; keep them in
  `backend/app/schemas.py`.
- Use `async` database sessions via `asyncpg`; inject via `Depends`.
- ML model training should run in a **background task** (`BackgroundTasks`) or
  a separate container worker — never block a request handler.
- Store trained model artifacts as files; record the path in `model_versions`.
- Return GeoJSON (`FeatureCollection`) for all geospatial responses.

---

## Frontend (React)

Entry: `frontend/src/main.jsx`.

### Map and interaction

- The map component is the central UI element. It uses **Leaflet** — check existing imports before adding another map library.
- Base imagery is served from the backend tile endpoint or an external WMS/XYZ
  tile service (see existing tile layer configuration).
- Drawing tools allow the user to draw polygons; after drawing, the user assigns
  a land cover class label.
- Prediction and uncertainty layers are rendered as raster tile layers on the map.

### Component structure (reference)

```
src/
├── components/
│   ├── ProjectWizard/   Multi-step new-project form (shown on first load
│   │                    or when no project is active)
│   ├── Map/             Main map container + layer control
│   ├── DrawingTools/    Polygon drawing + class label assignment
│   ├── ClassPanel/      Class selector and training sample list
│   ├── TrainPanel/      Train button, progress indicator, metrics
│   ├── ClassPickerModal/Assign class after drawing
│   ├── ConfirmDeleteModal/ Delete confirmation
│   └── Toast/           Transient notifications
├── context/             AppContext — active project, classes, samples, toasts
├── hooks/               useTrainStatus, useTrainingSamples, usePrediction, ...
├── services/            api.js — all fetch/axios calls to the backend
└── App.jsx              Single-page app, renders wizard or main layout
```

Keep state management simple (React Context or plain useState/useReducer);
do not introduce a state library unless explicitly requested.

### API integration rules

- All backend calls go through `src/services/api.js`; no direct `fetch` in components.
- When adding a new endpoint, add the corresponding function to `api.js` first,
  then use it in a hook or component.
- The active project id is stored in `AppContext` and passed as a query param or
  path segment to all project-scoped requests.
- After retraining completes, invalidate/refresh the prediction and uncertainty
  tile layers by bumping their URL cache-buster parameter.

---

## ML pipeline

ML code lives in `backend/app/ml/` or a dedicated `ml/` service.

### Model types

The model type is stored in `project.model_config` and governs which trainer is used:

| Family        | Options                                                | Best for                                                    |
| ------------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| Classical     | `random_forest`, `xgboost`, `svm`, `gradient_boosting` | Small datasets, tabular pixel features, fast iteration      |
| Deep learning | `unet`, `unet_resnet`                                  | Large annotated areas, spatial-context tasks (requires GPU) |

Start with `random_forest` as the default if no model is specified.

### Feature extraction

Features are derived from the project's `available_bands` and `enabled_indices`
configuration. **Training and prediction must use identical preprocessing** — any
change to feature extraction requires both pipelines to be updated together.

Common spectral indices and the bands they require:

| Index | Formula                               | Required bands        |
| ----- | ------------------------------------- | --------------------- |
| NDVI  | (NIR−R)/(NIR+R)                       | nir, red              |
| NDWI  | (G−NIR)/(G+NIR)                       | green, nir            |
| NDBI  | (SWIR−NIR)/(SWIR+NIR)                 | swir1, nir            |
| BSI   | ((SWIR+R)−(NIR+B))/((SWIR+R)+(NIR+B)) | swir1, red, nir, blue |
| EVI   | 2.5\*(NIR−R)/(NIR+6R−7.5B+1)          | nir, red, blue        |

Only compute indices whose required bands are listed in `project.available_bands`.
Fall back to raw band values if no indices are possible.

### Pipeline

- Training input: labeled polygons from PostGIS → rasterize over AOI → extract
  pixel features (bands + enabled indices) → fit model.
- Training output: serialized model saved to `ml/artifacts/model_<version>.pkl`;
  path and hyperparameters recorded in `model_versions`.
- Prediction output: classified raster → COG or tile-servable format → path
  stored in `predictions` table.
- Uncertainty: prediction entropy or margin sampling per pixel; stored in
  `uncertainty_tiles`.

---

## Project setup wizard

Shown on first load (no active project) and accessible via a "New Project" button.
It is a **multi-step form** that collects everything needed to initialise a project.

### Steps

**Step 1 — Project basics**

- Project name (required, unique)
- Description (optional, free text)
- Task type: `classification` (default) | `regression` ← only classification is
  implemented now; include the field for forward-compatibility

**Step 2 — Area of Interest**

- Draw a bounding polygon/rectangle on a mini-map, or type lat/lon bounds (min/max).
- The AOI is stored as a GeoJSON polygon in `projects.aoi_geometry` (EPSG:4326).
- Predictions are clipped to this AOI.

**Step 3 — Imagery source**

- Imagery URL template (XYZ/TMS format, e.g. OSM or a custom tile server).
- Available bands: multi-select checkboxes from a standard list
  (`blue`, `green`, `red`, `nir`, `red_edge`, `swir1`, `swir2`, `sar_vv`, `sar_vh`).
  Default: `blue, green, red` (RGB-only, works with any XYZ tile source).
- Enabled spectral indices: auto-filtered to only those computable from the
  selected bands (show as disabled + tooltip if bands are missing).
- Spatial resolution (metres per pixel, default `10`).

**Step 4 — Classes**

- Dynamic list of class entries: each has a **name** (text) and a **color** (color picker).
- Minimum 2 classes required.
- Pre-fill with sensible defaults based on task type (e.g. land cover defaults:
  Forest, Grassland, Cropland, Urban, Water, Bare Soil).
- User can add, remove, and reorder rows.

**Step 5 — Model**

- Model family: `Classical ML` | `Deep Learning`
- Model type within family (see ML pipeline section above).
- Key hyperparameters exposed per model type:
  - RF: n_estimators, max_depth
  - XGBoost: n_estimators, learning_rate, max_depth
  - SVM: kernel, C
  - U-Net: epochs, batch_size, encoder backbone
- All hyperparameters are optional with sensible defaults; store the full
  config as a JSON blob in `projects.model_config`.

### Behaviour rules

- On submit the wizard POSTs to `POST /projects`, which creates the project row
  and all class rows in a single transaction.
- The wizard result is stored in `AppContext.activeProject`; the main map/sidebar
  UI is only shown when an active project exists.
- A "Load existing project" option on the landing screen lets users resume work.
- Do **not** show the wizard again after a project is loaded; provide a settings
  icon to edit project config later (patch via `PATCH /projects/{id}`).
- Changing `available_bands` or `enabled_indices` after training invalidates
  existing model versions — warn the user but do not auto-delete them.

---

## Active learning

- After each prediction run, uncertainty scores are computed and stored.
- The frontend renders uncertainty as a heatmap tile layer on the map.
- High-uncertainty regions are highlighted to guide the user toward drawing
  new training polygons where the model is least confident.
- The uncertainty endpoint `/uncertainty/tiles/{z}/{x}/{y}` serves these tiles.

---

## Environment variables

Read from `.env` (never commit secrets). Expected variables:

```
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/lcmap
IMAGERY_SOURCE_URL=...          # base imagery tile URL or path
MODEL_ARTIFACTS_DIR=ml/artifacts
CORS_ORIGINS=http://localhost:3000
```

Add new variables to `.env.example` whenever you introduce them.

---

## Code style

- Python: follow existing formatting (Black + isort if configured); type-hint all
  function signatures; docstrings only for non-obvious functions.
- JavaScript: follow existing ESLint config; prefer functional components
  and hooks; no class components.
- SQL: run all schema changes through Alembic — no ad-hoc DDL.

---

## Files to read first

1. `docker-compose.yml` — understand service wiring and environment.
2. `backend/main.py` — see how routers and middleware are registered.
3. `backend/app/schemas.py` — understand data contracts.
4. `backend/app/ml/model.py` — understand current model and feature pipeline.
5. `frontend/src/services/api.js` — understand how frontend calls the backend.
6. `frontend/src/components/Map/` — understand map and layer setup.

---

## When unsure

- Prefer small, localized changes over broad refactors.
- Keep endpoints and components minimal — this is intentionally a simple app.
- Geospatial columns stay in EPSG:4326 unless you have an explicit reason.
- Training and prediction feature pipelines must stay in sync.
- Add or update tests alongside non-trivial logic changes.
