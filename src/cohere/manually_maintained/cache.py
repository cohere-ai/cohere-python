class CacheMixin:
    # Optional caching mixin with time-to-live. If cachetools is not installed, caching is disabled.
    try:
        from cachetools import TTLCache

        _cache = TTLCache(maxsize=100, ttl=60)
    except ImportError:
        _cache = None

    def _cache_get(self, key):
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key, value):
        if self._cache is not None:
            self._cache[key] = value
