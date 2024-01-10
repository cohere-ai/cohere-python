import pytest

from cohere.responses import EMBEDDINGS_FLOATS_RESPONSE_TYPE, Embeddings


@pytest.mark.asyncio
async def test_async_embed_basic(async_client):
    embs = await async_client.embed(model="small", texts=["co:here", "cohere"])
    assert len(embs) == 2
    assert isinstance(embs, Embeddings)
    assert embs.response_type == EMBEDDINGS_FLOATS_RESPONSE_TYPE
    assert embs.meta
    assert embs.meta["api_version"]
    assert embs.meta["api_version"]["version"]
