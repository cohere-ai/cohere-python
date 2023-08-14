import pytest

from cohere.responses import LogLikelihoods

TEST_MODEL = "command-light"


@pytest.mark.asyncio
async def test_basic_llh_async(async_client):
    resp = await async_client.loglikelihood(model=TEST_MODEL, prompt="co:here", completion="co:where?")
    assert isinstance(resp, LogLikelihoods)
    assert isinstance(resp.prompt_tokens[0].encoded, int)
    assert isinstance(resp.prompt_tokens[0].decoded, str)
    assert isinstance(resp.prompt_tokens[1].log_likelihood, float)

    assert resp.prompt_tokens[0].decoded == "<BOS_TOKEN>"
    assert resp.prompt_tokens[-1].decoded == "<EOP_TOKEN>"

    assert isinstance(resp.completion_tokens[0].encoded, int)
    assert isinstance(resp.completion_tokens[0].decoded, str)
    assert isinstance(resp.completion_tokens[0].log_likelihood, float)

    assert resp.completion_tokens[-1].decoded == "<EOS_TOKEN>"


@pytest.mark.asyncio
async def test_only_prompt_async_llh(async_client):
    resp = await async_client.loglikelihood(model=TEST_MODEL, prompt="co:here")
    assert isinstance(resp, LogLikelihoods)
    assert isinstance(resp.prompt_tokens[0].encoded, int)
    assert resp.completion_tokens is None
