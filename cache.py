import json
import hashlib
import logging
from typing import Optional, Any
import redis
from redis.connection import BlockingConnectionPool

logger = logging.getLogger(__name__)

class L1Cache:
    """基于 Redis 的精确匹配缓存"""
    def __init__(self, host: str, port: int, db: int, password: str = None,
                 ttl: int = 3600, max_connections: int = 10):
        self.pool = BlockingConnectionPool(
            host=host, port=port, db=db, password=password,
            max_connections=max_connections, decode_responses=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.ttl = ttl

    def _make_key(self, query: str) -> str:
        return f"l1:{hashlib.md5(query.encode('utf-8')).hexdigest()}"

    def get(self, query: str) -> Optional[str]:
        key = self._make_key(query)
        try:
            cached = self.client.get(key)
            if cached:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached
        except redis.RedisError as e:
            logger.error(f"Redis get error: {e}")
        return None

    def set(self, query: str, response: str) -> None:
        key = self._make_key(query)
        try:
            self.client.setex(key, self.ttl, response)
        except redis.RedisError as e:
            logger.error(f"Redis set error: {e}")

    def health_check(self) -> bool:
        try:
            return self.client.ping()
        except redis.RedisError:
            return False