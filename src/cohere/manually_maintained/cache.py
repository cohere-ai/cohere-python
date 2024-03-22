import time


class CacheMixin:
    # A simple in-memory cache with TTL (thread safe). This is used to cache tokenizers at the moment.
    _cache = dict()

    def _cache_get(self, key):
        val = self._cache.get(key)
        if val is None:
            return None
        expiry_timestamp, value = val
        if expiry_timestamp is None or expiry_timestamp > time.time():
            return value

        del self._cache[key]  # remove expired cache entry

    def _cache_set(self, key, value, ttl=60 * 60):
        expiry_timestamp = None
        if ttl is not None:
            expiry_timestamp = time.time() + ttl
        self._cache[key] = (expiry_timestamp, value)
