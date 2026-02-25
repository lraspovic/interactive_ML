# LC Mapper — Interactive ML for Remote Sensing

An interactive machine learning web application for remote sensing image classification. Draw training polygons on a map, train a classifier, and see live predictions — all in the browser.

![Stack](https://img.shields.io/badge/stack-React%20%7C%20FastAPI%20%7C%20PostGIS-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

The core loop:

1. **Create a project** — define your area of interest, imagery source, spectral bands, class scheme, and model type via a guided wizard.
2. **Search satellite imagery** — query Planetary Computer for Sentinel-2 scenes filtered by date and cloud cover.
3. **Draw training polygons** — use map drawing tools to annotate land cover classes directly on the imagery.
4. **Train a classifier** — pixel features (raw bands + spectral indices) are extracted from the COG assets and used to fit a scikit-learn model.
5. **Run prediction** — classify the current map viewport and display the result as a colour-coded overlay.
6. **Inspect uncertainty** — a per-pixel entropy heatmap guides active learning by highlighting where the model is least confident.

Applicable to vegetation mapping, urban analysis, crop monitoring, coastal/wetland assessment, burned-area detection, and more.

---

## Architecture

```
docker-compose
├── frontend/     React + Leaflet (port 3000)
├── backend/      FastAPI + rio-tiler + scikit-learn (port 8000)
└── db/           PostgreSQL 15 + PostGIS 3 (port 5432)
```

Satellite imagery is read directly from **Microsoft Planetary Computer** COG assets via HTTP range requests — no full scene downloads.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/lraspovic/interactive_ML.git
cd interactive_ML

# 2. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work out of the box for local dev)

# 3. Start everything
docker compose up --build

# 4. Open the app
open http://localhost:3000
```

To stop and remove all data:

```bash
docker compose down -v
```

---

## Environment variables

Defined in `.env` (see `.env.example`):

| Variable              | Default                                          | Description                                |
| --------------------- | ------------------------------------------------ | ------------------------------------------ |
| `DATABASE_URL`        | `postgresql+asyncpg://lcmap:lcmap@db:5432/lcmap` | PostGIS connection string                  |
| `IMAGERY_SOURCE_URL`  | —                                                | Optional custom tile source                |
| `MODEL_ARTIFACTS_DIR` | `ml/artifacts`                                   | Where trained model `.pkl` files are saved |
| `CORS_ORIGINS`        | `http://localhost:3000`                          | Allowed frontend origins                   |

---

## Supported models

| Family        | Types                                                  |
| ------------- | ------------------------------------------------------ |
| Classical ML  | `random_forest`, `xgboost`, `gradient_boosting`, `svm` |
| Deep learning | `unet`, `unet_resnet` _(planned)_                      |

Default: **Random Forest**. All hyperparameters are configurable via the project wizard.

---

## Supported spectral indices

Automatically computed from the bands selected in the project wizard:

| Index | Formula                               | Required bands        |
| ----- | ------------------------------------- | --------------------- |
| NDVI  | (NIR−R)/(NIR+R)                       | nir, red              |
| NDWI  | (G−NIR)/(G+NIR)                       | green, nir            |
| NDBI  | (SWIR−NIR)/(SWIR+NIR)                 | swir1, nir            |
| BSI   | ((SWIR+R)−(NIR+B))/((SWIR+R)+(NIR+B)) | swir1, red, nir, blue |
| EVI   | 2.5\*(NIR−R)/(NIR+6R−7.5B+1)          | nir, red, blue        |

---

## Development

**Backend only:**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend only:**

```bash
cd frontend
npm install
npm run dev
```

**Run backend tests:**

```bash
cd backend && pytest
```

**Apply database migrations:**

```bash
docker compose exec backend alembic upgrade head
```

---

## Project structure

```
backend/
├── app/
│   ├── ml/
│   │   ├── features.py     # Pixel feature extraction from COG assets
│   │   ├── trainer.py      # Model fitting + artifact serialisation
│   │   └── predictor.py    # Bbox prediction + PNG rendering
│   ├── routers/
│   │   ├── imagery.py      # STAC search + tile serving
│   │   ├── train.py        # Training endpoints + polygon split
│   │   ├── predict.py      # Prediction endpoints
│   │   ├── projects.py     # Project CRUD
│   │   └── training_samples.py
│   ├── models.py           # SQLAlchemy ORM models
│   └── schemas.py          # Pydantic request/response schemas
frontend/
├── src/
│   ├── components/
│   │   ├── Map/            # Leaflet map + drawing tools
│   │   ├── Sidebar/        # ClassPanel, TrainPanel, PredictPanel
│   │   ├── SatellitePanel/ # Scene search + band combos
│   │   └── ProjectWizard/  # 5-step new project form
│   ├── context/AppContext.jsx
│   └── services/api.js     # All backend calls
```

---

## License

MIT
