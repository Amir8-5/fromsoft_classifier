import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api import predict
from core.exceptions import FileTooLargeError, InvalidFileTypeError, ModelNotReadyError
from services.cache_service import CacheService
from services.model_service import ModelService

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + connect cache. Shutdown: close Redis."""
    print("Starting up — loading model and connecting to cache...")

    app.state.cache_service = CacheService()
    app.state.model_service = ModelService(
        device="cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    app.state.model_loaded = True

    print("Model loaded. API is ready.")
    yield

    # Teardown
    print("Shutting down — closing Redis connection...")
    await app.state.cache_service.redis.aclose()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FromSoftware Classifier API",
    description="Inference endpoint for game screenshots.",
    lifespan=lifespan,
)

# Attach limiter
app.state.limiter = limiter

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(FileTooLargeError)
async def file_too_large_handler(request: Request, exc: FileTooLargeError):
    return JSONResponse(
        status_code=413,
        content={"error": "Payload Too Large", "detail": str(exc)},
    )


@app.exception_handler(InvalidFileTypeError)
async def invalid_file_type_handler(request: Request, exc: InvalidFileTypeError):
    return JSONResponse(
        status_code=415,
        content={"error": "Unsupported Media Type", "detail": str(exc)},
    )


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    return JSONResponse(
        status_code=503,
        content={"error": "Service Unavailable", "detail": str(exc)},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api/v1")

# ── Lifecycle Probes ──────────────────────────────────────────────────────────
@app.get("/health", tags=["probes"])
def health():
    """Liveness probe — confirms the server process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["probes"])
def ready(request: Request):
    """Readiness probe — true once the model has been loaded on startup."""
    return {"model_loaded": getattr(request.app.state, "model_loaded", False)}


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
