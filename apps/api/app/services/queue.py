from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from apps.api.app.core.config import get_settings

QUEUE_NAME = "research-paper-jobs"


async def enqueue_job(payload: dict[str, Any]) -> None:
    redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        await redis.rpush(QUEUE_NAME, json.dumps(payload))
    finally:
        await redis.aclose()


async def dequeue_job(redis: Redis) -> dict[str, Any] | None:
    item = await redis.blpop(QUEUE_NAME, timeout=0)
    return json.loads(item[1]) if item else None