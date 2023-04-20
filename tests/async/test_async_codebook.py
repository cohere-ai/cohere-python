import pytest

from cohere.responses import Codebook


@pytest.mark.asyncio
async def test_async_multiple_codebook(async_client):
    compression_codebooks = {
        "32x": (96, 256, 8),
        "48x": (64, 256, 12),
        "64x": (48, 256, 16),
        "96x": (32, 256, 24),
    }
    for cb, (segments, num_centroids, length) in compression_codebooks.items():
        codebook = await async_client.codebook(
            model="multilingual-22-12",
            compression_codebook=cb,
        )
        assert isinstance(codebook, Codebook)
        assert isinstance(codebook.codebook[0], list)
        assert isinstance(codebook.codebook[0][0], list)
        assert len(codebook.codebook) == segments
        assert len(codebook.codebook[0]) == num_centroids
        assert len(codebook.codebook[0][0]) == length
        assert codebook.meta
        assert codebook.meta["api_version"]["version"]
        assert codebook.meta["api_version"]


@pytest.mark.asyncio
async def test_async_codebook_default(async_client):
    codebook = await async_client.codebook(model="multilingual-22-12")
    assert len(codebook) == 96
    assert isinstance(codebook, Codebook)
    assert codebook.meta
    assert codebook.meta["api_version"]
    assert codebook.meta["api_version"]["version"]
