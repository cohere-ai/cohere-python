class CacheMixin:
    # Optional caching mixin with time-to-live. If cachetools is not installed, caching is disabled.
    cache = None
    try:
        from cachetools import TTLCache

        cache = TTLCache(maxsize=100, ttl=60)
    except ImportError:
        pass

    def cache_get(self, key):
        if self.cache is None:
            return None
        return self.cache.get(key)

    def cache_set(self, key, value):
        if self.cache is not None:
            self.cache[key] = value
