import pytest

import cohere


@pytest.mark.asyncio
async def test_tokenize(client):
    tokens = await client.tokenize("tokenize me!")
    assert isinstance(tokens.tokens, list)
    assert isinstance(tokens.token_strings, list)
    assert isinstance(len(tokens), int)
    assert len(tokens.tokens) == len(tokens)
    assert len(tokens.token_strings) == len(tokens)


@pytest.mark.asyncio
async def test_invalid_text(client):
    with pytest.raises(cohere.CohereError):
        await client.tokenize("")
