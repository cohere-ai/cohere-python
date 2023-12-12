import pytest

import cohere


@pytest.mark.asyncio
async def test_async_enter():
    cli = cohere.AsyncClient()
    async with cli as co:
        await co.generate(model="command-light", prompt="co:here", max_tokens=1)
    assert cli._backend._session is None


@pytest.mark.asyncio
async def test_async_enter_invalid_key(monkeypatch):
    api_key = ""
    monkeypatch.setenv("CO_API_KEY", api_key)
    cli = cohere.AsyncClient(api_key=api_key)
    with pytest.raises(cohere.CohereError) as exc:
        async with cli as co:
            pass


@pytest.mark.asyncio
async def test_async_404(async_client):
    with pytest.raises(cohere.CohereError) as exc:
        await async_client._request("/test-404")
