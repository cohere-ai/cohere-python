import os
import unittest

import httpx

import cohere


class TestEnvironmentProxyMount(unittest.TestCase):
    """Regression tests for #817: the default client must honour proxy env vars."""

    def test_default_client_mounts_env_proxy(self) -> None:
        os.environ["HTTP_PROXY"] = "http://proxy.example.com:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.example.com:8080"
        try:
            client = cohere.ClientV2(api_key="n/a")
        finally:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        inner = client._client_wrapper.httpx_client.httpx_client
        proxy_transports = [
            transport
            for transport in inner._mounts.values()
            if getattr(getattr(transport, "_pool", None), "_proxy_url", None) is not None
        ]
        self.assertTrue(proxy_transports, "default client should mount an env proxy transport")

    def test_no_proxy_bypass_mount(self) -> None:
        os.environ["NO_PROXY"] = "localhost,internal.example.com"
        try:
            client = cohere.ClientV2(api_key="n/a")
        finally:
            os.environ.pop("NO_PROXY", None)
        inner = client._client_wrapper.httpx_client.httpx_client
        bypass = [pattern for pattern, transport in inner._mounts.items() if transport is None]
        self.assertTrue(bypass, "NO_PROXY should produce a bypass (None) mount")

    def test_explicit_httpx_client_is_untouched(self) -> None:
        os.environ["HTTP_PROXY"] = "http://proxy.example.com:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.example.com:8080"
        explicit = httpx.Client()
        before = dict(explicit._mounts)
        try:
            cohere.Client(api_key="n/a", httpx_client=explicit)
        finally:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        after = dict(explicit._mounts)
        self.assertEqual(before, after, "explicit httpx_client must not be modified")

    def test_proxied_vs_bypass_dispatch(self) -> None:
        os.environ["HTTP_PROXY"] = "http://proxy.example.com:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.example.com:8080"
        os.environ["NO_PROXY"] = "internal.example.com"
        try:
            client = cohere.ClientV2(api_key="n/a")
        finally:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
                os.environ.pop(key, None)
        inner = client._client_wrapper.httpx_client.httpx_client

        def proxy_url(transport):
            return getattr(getattr(transport, "_pool", None), "_proxy_url", None)

        proxied = inner._transport_for_url(httpx.URL("https://api.cohere.com/v1/x"))
        bypass = inner._transport_for_url(httpx.URL("https://internal.example.com/x"))
        self.assertIsNotNone(proxy_url(proxied), "proxied host must route through the proxy")
        self.assertIsNone(proxy_url(bypass), "NO_PROXY host must bypass the proxy")

    def test_async_client_mounts_env_proxy(self) -> None:
        os.environ["HTTP_PROXY"] = "http://proxy.example.com:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy.example.com:8080"
        try:
            client = cohere.AsyncClient(api_key="n/a")
        finally:
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        inner = client._client_wrapper.httpx_client.httpx_client
        async_proxy = [
            transport
            for transport in inner._mounts.values()
            if type(transport).__name__ == "AsyncHTTPTransport"
            and getattr(getattr(transport, "_pool", None), "_proxy_url", None) is not None
        ]
        self.assertTrue(async_proxy, "async default client should mount an env proxy transport")
