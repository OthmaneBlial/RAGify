import json
import logging
import hashlib
from typing import Any, Dict, List, Optional
from functools import lru_cache
import time
from dataclasses import dataclass
from enum import Enum

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from cachetools import TTLCache
from .config import settings

logger = logging.getLogger(__name__)


def _stable_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class CacheType(Enum):
    """Types of cache entries."""

    EMBEDDING = "embedding"
    SEARCH_RESULT = "search_result"
    DOCUMENT_CONTENT = "document_content"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: float = None
    cache_type: CacheType = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "cache_type": self.cache_type.value if self.cache_type else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        data_copy = data.copy()
        if data_copy.get("cache_type"):
            data_copy["cache_type"] = CacheType(data_copy["cache_type"])
        return cls(**data_copy)


class CacheManager:
    """Unified cache manager with Redis and in-memory LRU support."""

    def __init__(self):
        self.redis_client = None
        self._redis_disabled = False
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour default TTL
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis client if available."""
        if redis and hasattr(settings, "redis_url") and settings.redis_url:
            try:
                self.redis_client = redis.from_url(settings.redis_url)
                self._redis_disabled = False
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.redis_client = None
                self._redis_disabled = True
        else:
            logger.info("Redis not available, using in-memory cache only")
            self._redis_disabled = True

    def _disable_redis(self, error: Exception = None):
        """Disable Redis usage after repeated failures."""
        if self.redis_client:
            if error:
                logger.warning(f"Disabling Redis cache after error: {error}")
            else:
                logger.warning("Disabling Redis cache")
        self.redis_client = None
        self._redis_disabled = True

    async def get(self, key: str, cache_type: CacheType = None) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            cache_type: Type of cache entry

        Returns:
            Cached value or None
        """
        # Try Redis first
        if self.redis_client:
            try:
                redis_key = f"{cache_type.value}:{key}" if cache_type else key
                data = await self.redis_client.get(redis_key)
                if data:
                    entry_dict = json.loads(data.decode("utf-8"))
                    entry = CacheEntry.from_dict(entry_dict)
                    if not entry.is_expired:
                        logger.debug(f"Cache hit (Redis): {key}")
                        return entry.value
                    else:
                        # Remove expired entry
                        await self.redis_client.delete(redis_key)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                self._disable_redis(e)

        # Try memory cache
        memory_key = f"{cache_type.value}:{key}" if cache_type else key
        if memory_key in self.memory_cache:
            entry = self.memory_cache[memory_key]
            if not entry.is_expired:
                logger.debug(f"Cache hit (Memory): {key}")
                return entry.value
            else:
                # Remove expired entry
                del self.memory_cache[memory_key]

        logger.debug(f"Cache miss: {key}")
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: CacheType = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_type: Type of cache entry

        Returns:
            Success status
        """
        entry = CacheEntry(key=key, value=value, ttl=ttl, cache_type=cache_type)

        success = True

        # Set in Redis
        if self.redis_client:
            try:
                redis_key = f"{cache_type.value}:{key}" if cache_type else key
                data = json.dumps(entry.to_dict())
                await self.redis_client.set(redis_key, data, ex=ttl)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                success = False
                self._disable_redis(e)

        # Set in memory cache
        memory_key = f"{cache_type.value}:{key}" if cache_type else key
        self.memory_cache[memory_key] = entry

        return success

    async def delete(self, key: str, cache_type: CacheType = None) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key
            cache_type: Type of cache entry

        Returns:
            Success status
        """
        success = True

        # Delete from Redis
        if self.redis_client:
            try:
                redis_key = f"{cache_type.value}:{key}" if cache_type else key
                await self.redis_client.delete(redis_key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
                success = False
                self._disable_redis(e)

        # Delete from memory cache
        memory_key = f"{cache_type.value}:{key}" if cache_type else key
        if memory_key in self.memory_cache:
            del self.memory_cache[memory_key]

        return success

    async def invalidate_by_pattern(
        self, pattern: str, cache_type: CacheType = None
    ) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Pattern to match (Redis pattern syntax)
            cache_type: Type of cache entry

        Returns:
            Number of entries invalidated
        """
        invalidated = 0

        # Invalidate in Redis
        if self.redis_client:
            try:
                redis_pattern = (
                    f"{cache_type.value}:{pattern}" if cache_type else pattern
                )
                cursor = b"0"
                keys = []
                while True:
                    cursor, batch = await self.redis_client.scan(
                        cursor=cursor, match=redis_pattern, count=1000
                    )
                    keys.extend(batch)
                    if cursor == b"0":
                        break
                if keys:
                    await self.redis_client.delete(*keys)
                    invalidated += len(keys)
            except Exception as e:
                logger.error(f"Redis pattern delete error: {e}")

        # Invalidate in memory cache
        memory_pattern = f"{cache_type.value}:{pattern}" if cache_type else pattern
        keys_to_delete = [
            k
            for k in self.memory_cache.keys()
            if self._matches_pattern(k, memory_pattern)
        ]
        for key in keys_to_delete:
            del self.memory_cache[key]
        invalidated += len(keys_to_delete)

        return invalidated

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for memory cache."""
        # Convert Redis-style pattern to simple wildcard
        import fnmatch

        return fnmatch.fnmatch(key, pattern.replace("*", "*"))

    async def invalidate_knowledge_base(self, kb_id: str) -> int:
        """
        Invalidate all cache entries related to a knowledge base.

        Args:
            kb_id: Knowledge base ID

        Returns:
            Number of entries invalidated
        """
        patterns = [
            f"kb:{kb_id}:*",
            f"embedding:{kb_id}:*",
            f"search:{kb_id}:*",
            f"document:{kb_id}:*",
        ]

        total_invalidated = 0
        for pattern in patterns:
            total_invalidated += await self.invalidate_by_pattern(pattern)

        logger.info(
            f"Invalidated {total_invalidated} cache entries for knowledge base {kb_id}"
        )
        return total_invalidated

    async def clear_all(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            Success status
        """
        success = True

        # Clear Redis
        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
                success = False

        # Clear memory cache
        self.memory_cache.clear()

        return success

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_maxsize": self.memory_cache.maxsize,
            "redis_available": self.redis_client is not None,
        }

        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update(
                    {
                        "redis_used_memory": info.get("used_memory"),
                        "redis_total_keys": await self.redis_client.dbsize(),
                    }
                )
            except Exception as e:
                logger.error(f"Redis stats error: {e}")

        return stats

    async def close(self):
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions
async def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding for text."""
    return await cache_manager.get(
        f"embedding:{_stable_key(text)}", CacheType.EMBEDDING
    )


async def set_cached_embedding(text: str, embedding: List[float], ttl: int = 3600):
    """Cache embedding for text."""
    await cache_manager.set(
        f"embedding:{_stable_key(text)}", embedding, ttl, CacheType.EMBEDDING
    )


async def get_cached_search_result(query: str, kb_id: str) -> Optional[Dict[str, Any]]:
    """Get cached search result."""
    return await cache_manager.get(
        f"search:{kb_id}:{_stable_key(query)}", CacheType.SEARCH_RESULT
    )


async def set_cached_search_result(
    query: str, kb_id: str, result: Dict[str, Any], ttl: int = 1800
):
    """Cache search result."""
    await cache_manager.set(
        f"search:{kb_id}:{_stable_key(query)}", result, ttl, CacheType.SEARCH_RESULT
    )


# LRU cache for frequently accessed data
@lru_cache(maxsize=512)
def _frequent_data_internal(key: str) -> Any:
    return None


def set_frequent_data(key: str, value: Any):
    _frequent_data_internal.cache_clear()

    def loader(k=key, v=value):
        return v

    globals()["_frequent_data_internal"] = lru_cache(maxsize=512)(lambda k: value)


def get_frequent_data(key: str) -> Optional[Any]:
    return _frequent_data_internal(key)
