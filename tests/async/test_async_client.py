import pytest

import cohere


@pytest.mark.asyncio
async def test_async_enter():
    cli = cohere.AsyncClient()
    async with cli as co:
        await co.generate(model="medium", prompt="co:here", max_tokens=1)
    assert cli._backend._session is None


@pytest.mark.asyncio
async def test_async_enter_invalid_key():
    cli = cohere.AsyncClient(api_key="invalid")
    with pytest.raises(cohere.CohereError) as exc:
        async with cli as co:
            pass
    assert "invalid API key" in str(exc)


@pytest.mark.asyncio
async def test_async_404(async_client):
    with pytest.raises(cohere.CohereError) as exc:
        await async_client._request("/test-404")
