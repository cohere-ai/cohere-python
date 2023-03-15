import pytest

import cohere
from cohere.responses import Detokenization


@pytest.mark.asyncio
async def test_tokenize(async_client):
    tokens = await async_client.tokenize("tokenize me!")
    assert isinstance(tokens.tokens, list)
    assert isinstance(tokens.token_strings, list)
    assert isinstance(len(tokens), int)
    assert len(tokens.tokens) == len(tokens)
    assert len(tokens.token_strings) == len(tokens)
    assert tokens.meta
    assert tokens.meta["api_version"]
    assert tokens.meta["api_version"]["version"]


@pytest.mark.asyncio
async def test_invalid_text(async_client):
    with pytest.raises(cohere.CohereError):
        await async_client.tokenize("")


@pytest.mark.asyncio
async def test_detokenize(async_client):
    detokenized = await async_client.detokenize([10104, 12221, 974, 514, 34])
    assert detokenized == Detokenization("detokenize me!")
    assert detokenized.meta
    assert detokenized.meta["api_version"]
    assert detokenized.meta["api_version"]["version"]

    detokenizeds = await async_client.batch_detokenize([[10104, 12221, 974, 514, 34]] * 3)
    assert detokenizeds == [Detokenization("detokenize me!")] * 3

    detokenized = await async_client.detokenize([])
    assert detokenized == Detokenization("")
