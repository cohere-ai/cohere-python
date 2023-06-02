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
async def test_model_param_tokenization(async_client):
    # Use tuples to be hashable (for set equality)
    medium_res = tuple((await async_client.tokenize("Hello world!", model="medium")).tokens)
    medium_res_batch = [
        tuple(x.tokens) for x in await async_client.batch_tokenize(["Hello world!"] * 3, model="medium")
    ]

    command_res = tuple((await async_client.tokenize("Hello world!", model="command")).tokens)
    command_res_batch = [
        tuple(x.tokens) for x in await async_client.batch_tokenize(["Hello world!"] * 3, model="command")
    ]

    # Assert that the result of one tokenization is the same as the result of a batch tokenization (model passed)
    # and that the result of a tokenization with one model is different from another model
    assert set([medium_res]) == set(medium_res_batch)
    assert set([command_res]) == set(command_res_batch)
    assert medium_res != command_res


@pytest.mark.asyncio
async def test_invalid_text(async_client):
    with pytest.raises(cohere.CohereError):
        await async_client.tokenize("")


@pytest.mark.asyncio
async def test_detokenize(async_client):
    detokenized = await async_client.detokenize([10104, 12221, 974, 514, 34], model="base")
    assert detokenized == Detokenization("detokenize me!")
    assert detokenized.meta
    assert detokenized.meta["api_version"]
    assert detokenized.meta["api_version"]["version"]

    detokenizeds = await async_client.batch_detokenize([[10104, 12221, 974, 514, 34]] * 3, model="base")
    assert detokenizeds == [Detokenization("detokenize me!")] * 3

    detokenized = await async_client.detokenize([])
    assert detokenized == Detokenization("")


@pytest.mark.asyncio
async def test_model_param_detokenization(async_client):
    medium_detokenized = (await async_client.detokenize([10104, 12221, 974, 514, 34], model="medium")).text
    medium_detokenized_batch = [
        x.text for x in await async_client.batch_detokenize([[10104, 12221, 974, 514, 34]] * 3, model="medium")
    ]

    command_detokenized = (await async_client.detokenize([10104, 12221, 974, 514, 34], model="command")).text
    command_detokenized_batch = [
        x.text for x in await async_client.batch_detokenize([[10104, 12221, 974, 514, 34]] * 3, model="command")
    ]

    assert set([medium_detokenized]) == set(medium_detokenized_batch)
    assert set([command_detokenized]) == set(command_detokenized_batch)
    assert medium_detokenized != command_detokenized
