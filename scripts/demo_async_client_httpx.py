"""
Run chat twice: once with the SDK default async HTTP stack (HttpxAiohttpClient),
once with an explicit httpx.AsyncClient.

Requires CO_API_KEY in the environment. Optional: CO_MODEL (defaults to command-r-plus).
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

from cohere import AsyncClient


def _api_key() -> str:
    key = os.getenv("CO_API_KEY")
    if not key:
        print("Set CO_API_KEY to your Cohere API key.", file=sys.stderr)
        sys.exit(1)
    return key


def _model() -> str:
    return os.getenv("CO_MODEL", "command-a-03-2025")


async def chat_default_httpx_aiohttp() -> None:
    """Uses the SDK default (aiohttp-backed HttpxAiohttpClient)."""
    async with AsyncClient(api_key=_api_key()) as client:
        inner = client._client_wrapper.httpx_client.httpx_client
        print(f"default inner client: {type(inner).__module__}.{type(inner).__name__}")
        response = await client.chat(message="Hello", model=_model())
        print("default — reply:", (response.text or "")[:200])


async def chat_custom_httpx() -> None:
    """Uses a caller-provided httpx.AsyncClient."""
    custom = httpx.AsyncClient(timeout=120.0)
    try:
        async with AsyncClient(api_key=_api_key(), httpx_client=custom) as client:
            inner = client._client_wrapper.httpx_client.httpx_client
            print(f"custom inner client: {type(inner).__module__}.{type(inner).__name__}")
            assert inner is custom
            response = await client.chat(message="Hello", model=_model())
            print("custom — reply:", (response.text or "")[:200])
    finally:
        # Context manager already closed `custom` when using httpx_client=; this is a no-op if closed.
        if not custom.is_closed:
            await custom.aclose()


async def main() -> None:
    await chat_default_httpx_aiohttp()
    await chat_custom_httpx()


if __name__ == "__main__":
    asyncio.run(main())
