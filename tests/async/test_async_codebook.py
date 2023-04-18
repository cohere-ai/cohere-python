import pytest

from cohere.responses import Codebook


@pytest.mark.asyncio
async def test_async_codebook_basic(async_client):
    codebook = await async_client.codebook(model="multilingual-22-12")
    assert len(codebook) == 96
    assert isinstance(codebook, Codebook)
    assert codebook.meta
    assert codebook.meta["api_version"]
    assert codebook.meta["api_version"]["version"]
