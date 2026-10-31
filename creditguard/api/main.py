"""FastAPI application.

Run locally:  uvicorn creditguard.api.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from creditguard import __version__, config
from creditguard.api.model_service import ModelNotLoadedError, ModelService
from creditguard.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logger = logging.getLogger("creditguard.api")


def create_app(model_path: Path | None = None, meta_path: Path | None = None) -> FastAPI:
    service = ModelService(model_path or config.MODEL_PATH, meta_path or config.META_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load()  # once per process, never per request
        if service.loaded:
            logger.info("Model loaded: %s", service.version)
        else:
            logger.warning("Model artifacts not found at %s", service.model_path)
        yield

    app = FastAPI(
        title="CreditGuard API",
        version=__version__,
        description="Credit card default risk scoring for analyst decision support.",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/")
    def root() -> dict:
        return {"service": "CreditGuard API", "version": __version__, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ok = service.loaded
        return HealthResponse(status="ok" if ok else "degraded", model_loaded=ok)

    @app.get("/model-info")
    def model_info() -> dict:
        if not service.loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        keys = [
            "project_version", "trained_at_utc", "trained_on_synthetic_data", "selected_model",
            "threshold", "base_rate", "cost_assumptions", "operating_points", "split", "cv_comparison", "test_metrics",
            "calibration", "fairness_audit", "permutation_importance",
        ]
        return {k: service.meta.get(k) for k in keys}

    @app.get("/operating-points")
    def operating_points() -> dict:
        if not service.loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        return {
            "default_threshold": service.threshold,
            "operating_points": service.meta.get("operating_points", []),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: CustomerFeatures,
        threshold: float | None = Query(
            default=None, ge=0.01, le=0.99,
            description="Optional flag cut-off. Defaults to the model's cost-based threshold.",
        ),
    ) -> PredictionResponse:
        try:
            result = service.predict(payload.model_dump(), threshold)
        except ModelNotLoadedError:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        return PredictionResponse(**result)

    return app


app = create_app()
