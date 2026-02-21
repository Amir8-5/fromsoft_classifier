import json
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 86400  # 24 hours


class CacheService:
    def __init__(self):
        self.redis = redis.from_url(
            "redis://localhost:6379",
            decode_responses=True,
        )

    async def get_cached_prediction(self, image_hash: str) -> dict | None:
        """
        Return a cached prediction dict for the given image hash, or None on miss / Redis offline.
        Fail-open: a ConnectionError is logged as a warning and None is returned so the
        request continues to the model inference path.
        """
        try:
            raw = await self.redis.get(image_hash)
            if raw is not None:
                return json.loads(raw)
            return None
        except redis.exceptions.ConnectionError:
            logger.warning("Redis unavailable — skipping cache read, proceeding to inference.")
            return None

    async def set_cached_prediction(self, image_hash: str, prediction: dict) -> None:
        """
        Store a prediction dict under the image hash with a 24-hour TTL.
        Fail-open: a ConnectionError is logged as a warning and silently ignored so the
        response is still returned to the caller.
        """
        try:
            await self.redis.set(image_hash, json.dumps(prediction), ex=_CACHE_TTL_SECONDS)
        except redis.exceptions.ConnectionError:
            logger.warning("Redis unavailable — prediction result will not be cached.")
