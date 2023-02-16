import pytest

from cohere.responses import Embeddings


@pytest.mark.asyncio
async def test_embed_basic(client):
    embs = await client.embed(model="small", texts=["co:here", "cohere"])
    assert len(embs) == 2
    assert isinstance(embs, Embeddings)
