from typing import Any, Dict, List, Optional, Tuple, Union
import time
from sklearn.linear_model import LogisticRegression
import hashlib
import json


class CacheMixin:
    # A simple in-memory cache with TTL (thread safe). This is used to cache tokenizers at the moment.
    _cache: Dict[str, Tuple[Optional[float], Any]] = dict()

    def _cache_get(self, key: str) -> Any:
        val = self._cache.get(key)
        if val is None:
            return None
        expiry_timestamp, value = val
        if expiry_timestamp is None or expiry_timestamp > time.time():
            return value

        del self._cache[key]  # remove expired cache entry

    def _cache_set(self, key: str, value: Any, ttl: int = 60 * 60) -> None:
        expiry_timestamp = None
        if ttl is not None:
            expiry_timestamp = time.time() + ttl
        self._cache[key] = (expiry_timestamp, value)


class ModelCache:
    """
    A cache system for storing LogisticRegression models where they key is a list of string pairs.
    Handles serialization, key hashing, and memory management.
    """
    # A simple in-memory cache with TTL (thread safe). This is used to cache LR models at the moment.
    _cache: Dict[str, Tuple[Optional[float], LogisticRegression]] = dict()
            
    def _hash_key(self, key: List[Tuple[str, str]]) -> str:
        """
        Convert a list of string pairs into a unique hash string.
        
        Args:
            key: List of tuples, each containing a pair of strings
            
        Returns:
            str: Hash string representing the key
        """
        # Sort the key to ensure consistent hashing regardless of order
        sorted_key = sorted(key)
        
        # Convert to JSON string for consistent serialization
        key_str = json.dumps(sorted_key)
        
        # Create hash
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _validate_key(self, key: List[Tuple[str, str]]) -> None:
        """
        Validate that the key is in the correct format.
        
        Args:
            key: List of tuples to validate
            
        Raises:
            ValueError: If key format is invalid
        """
        if not isinstance(key, list):
            raise ValueError("Key must be a list")
            
        for pair in key:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Each key element must be a tuple of length 2")
            if not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise ValueError("Each tuple must contain a pair of strings")
    
    def set(self, key: List[Tuple[str, int]], model: LogisticRegression, ttl: int = 60 * 60) -> None:
        """
        Store a model in the cache.
        
        Args:
            key: List of (string, integer) tuples
            model: Trained LogisticRegression model to store
        """
        self._validate_key(key)
        
        hashed_key = self._hash_key(key)

        expiry_timestamp = None
        if ttl is not None:
            expiry_timestamp = time.time() + ttl
        
        self._cache[hashed_key] = (expiry_timestamp, model)
    
    def get(self, key: List[Tuple[str, int]]) -> Union[LogisticRegression, None]:
        """
        Retrieve a model from the cache.
        
        Args:
            key: List of (string, integer) tuples
            
        Returns:
            LogisticRegression model if found, None if not in cache
        """
        self._validate_key(key)
        hashed_key = self._hash_key(key)
        val = self._cache.get(hashed_key)
        if val is None:
            return None
        expiry_timestamp, value = val
        if expiry_timestamp is None or expiry_timestamp > time.time():
            return value

        del self._cache[hashed_key]  # remove expired cache entry
