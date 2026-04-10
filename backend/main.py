"""FastAPI application for object detection inference."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import detect, health, models
from backend.services.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_manager = ModelManager()
    yield
    app.state.model_manager.unload()


app = FastAPI(
    title="Object Detection API",
    description="OpenAI-compatible object detection inference service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(detect.router)
