from fastapi import APIRouter, File, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.file_service import FileService

# Mirror the same limiter instance used in main.py
# (SlowAPI resolves limits via app.state.limiter — this local reference is used
#  only for the @limiter.limit decorator syntax)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


@router.post("/predict", tags=["inference"])
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Classify a FromSoftware game screenshot.

    - Validates MIME type (jpeg/png) and size (≤ 5 MB).
    - Returns cached result immediately if the exact image was seen before.
    - Otherwise runs inference and caches the result for 24 hours.

    Rate limit: 10 requests / minute per IP.
    """
    model_service = request.app.state.model_service
    cache_service = request.app.state.cache_service

    # Step 1: Validate file and compute hash
    image_hash = await FileService.validate_and_hash(file)

    # Step 2: Check cache
    cached = await cache_service.get_cached_prediction(image_hash)
    if cached is not None:
        return {**cached, "cached": True}

    # Step 3: Read bytes (file pointer already reset by validate_and_hash)
    image_bytes = await file.read()

    # Step 4: Run inference
    result = model_service.predict(image_bytes)

    # Step 5: Cache result
    await cache_service.set_cached_prediction(image_hash, result)

    return {**result, "cached": False}
