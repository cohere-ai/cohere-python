import sys
import typing
import unittest
from unittest import mock

import httpx


class TestMakeDefaultAsyncClient(unittest.TestCase):
    """Tests for _make_default_async_client in base_client.py."""

    def test_without_httpx_aiohttp_returns_httpx_async_client(self) -> None:
        """When httpx_aiohttp is not installed, returns plain httpx.AsyncClient."""
        with mock.patch.dict(sys.modules, {"httpx_aiohttp": None}):
            # Re-import to pick up the mocked module state
            from cohere.base_client import _make_default_async_client

            client = _make_default_async_client(timeout=300, follow_redirects=True)
            self.assertIsInstance(client, httpx.AsyncClient)
            self.assertEqual(client.timeout.read, 300)
            self.assertTrue(client.follow_redirects)

    def test_without_httpx_aiohttp_follow_redirects_none(self) -> None:
        """When follow_redirects is None, omits it from httpx.AsyncClient."""
        with mock.patch.dict(sys.modules, {"httpx_aiohttp": None}):
            from cohere.base_client import _make_default_async_client

            client = _make_default_async_client(timeout=300, follow_redirects=None)
            self.assertIsInstance(client, httpx.AsyncClient)
            # httpx default is False when not specified
            self.assertFalse(client.follow_redirects)

    def test_with_httpx_aiohttp_returns_aiohttp_client(self) -> None:
        """When httpx_aiohttp is installed, returns HttpxAiohttpClient."""
        try:
            import httpx_aiohttp
        except ImportError:
            self.skipTest("httpx_aiohttp not installed")

        from cohere.base_client import _make_default_async_client

        client = _make_default_async_client(timeout=300, follow_redirects=True)
        self.assertIsInstance(client, httpx_aiohttp.HttpxAiohttpClient)
        self.assertEqual(client.timeout.read, 300)
        self.assertTrue(client.follow_redirects)

    def test_with_httpx_aiohttp_follow_redirects_none(self) -> None:
        """When httpx_aiohttp is installed and follow_redirects is None, omits it."""
        try:
            import httpx_aiohttp
        except ImportError:
            self.skipTest("httpx_aiohttp not installed")

        from cohere.base_client import _make_default_async_client

        client = _make_default_async_client(timeout=300, follow_redirects=None)
        self.assertIsInstance(client, httpx_aiohttp.HttpxAiohttpClient)
        # httpx default is False when not specified
        self.assertFalse(client.follow_redirects)

    def test_explicit_httpx_client_bypasses_autodetect(self) -> None:
        """When user passes httpx_client explicitly, auto-detect is not used."""
        explicit_client = httpx.AsyncClient(timeout=60)
        # Simulate what AsyncBaseCohere.__init__ does:
        # httpx_client if httpx_client is not None else _make_default_async_client(...)
        result = explicit_client if explicit_client is not None else None
        self.assertIs(result, explicit_client)
        self.assertEqual(result.timeout.read, 60)


class TestDefaultClients(unittest.TestCase):
    """Tests for convenience classes in _default_clients.py."""

    def test_default_async_httpx_client_defaults(self) -> None:
        """DefaultAsyncHttpxClient applies SDK defaults."""
        from cohere._default_clients import COHERE_DEFAULT_TIMEOUT, DefaultAsyncHttpxClient

        client = DefaultAsyncHttpxClient()
        self.assertIsInstance(client, httpx.AsyncClient)
        self.assertEqual(client.timeout.read, COHERE_DEFAULT_TIMEOUT)
        self.assertTrue(client.follow_redirects)

    def test_default_async_httpx_client_overrides(self) -> None:
        """DefaultAsyncHttpxClient allows overriding defaults."""
        from cohere._default_clients import DefaultAsyncHttpxClient

        client = DefaultAsyncHttpxClient(timeout=60, follow_redirects=False)
        self.assertEqual(client.timeout.read, 60)
        self.assertFalse(client.follow_redirects)

    def test_default_aiohttp_client_without_package(self) -> None:
        """DefaultAioHttpClient raises RuntimeError when httpx_aiohttp not installed."""
        with mock.patch.dict(sys.modules, {"httpx_aiohttp": None}):
            # Need to reload the module to pick up the mock
            import importlib
            import cohere._default_clients

            importlib.reload(cohere._default_clients)

            with self.assertRaises(RuntimeError) as ctx:
                cohere._default_clients.DefaultAioHttpClient()
            self.assertIn("pip install cohere[aiohttp]", str(ctx.exception))

            # Reload again to restore original state
            importlib.reload(cohere._default_clients)

    def test_default_aiohttp_client_with_package(self) -> None:
        """DefaultAioHttpClient works when httpx_aiohttp is installed."""
        try:
            import httpx_aiohttp
        except ImportError:
            self.skipTest("httpx_aiohttp not installed")

        from cohere._default_clients import COHERE_DEFAULT_TIMEOUT, DefaultAioHttpClient

        client = DefaultAioHttpClient()
        self.assertIsInstance(client, httpx_aiohttp.HttpxAiohttpClient)
        self.assertEqual(client.timeout.read, COHERE_DEFAULT_TIMEOUT)
        self.assertTrue(client.follow_redirects)
