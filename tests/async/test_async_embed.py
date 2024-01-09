import pytest

from cohere.responses import Embeddings


@pytest.mark.asyncio
async def test_async_embed_basic(async_client):
    embs = await async_client.embed(model="small", texts=["co:here", "cohere"])
    assert len(embs) == 2
    assert isinstance(embs, Embeddings)
    assert embs.response_type == "embeddings_floats"
    assert embs.meta
    assert embs.meta["api_version"]
    assert embs.meta["api_version"]["version"]
