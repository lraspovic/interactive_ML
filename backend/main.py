import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.classes import router as classes_router
from app.routers.imagery import router as imagery_router
from app.routers.predict import router as predict_router
from app.routers.projects import router as projects_router
from app.routers.train import router as train_router
from app.routers.training_samples import router as training_samples_router

app = FastAPI(title="Interactive ML - Land Cover Mapping")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classes_router)
app.include_router(imagery_router)
app.include_router(predict_router)
app.include_router(projects_router)
app.include_router(train_router)
app.include_router(training_samples_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend"}
