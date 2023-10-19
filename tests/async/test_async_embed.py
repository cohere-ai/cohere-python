import pytest

from cohere.responses import Embeddings

# 1. for oci
MODEL = "cohere.embed-english-light-v2.0"
# 2. for cohere
# MODEL = "small"

@pytest.mark.asyncio
async def test_async_embed_basic(async_client):
    embs = await async_client.embed(model=MODEL, texts=["co:here", "cohere"])
    assert len(embs) == 2
    assert isinstance(embs, Embeddings)
    assert embs.meta
    assert embs.meta["api_version"]
    assert embs.meta["api_version"]["version"]
