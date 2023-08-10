from cohere.responses import LogLikelihoods

TEST_MODEL = "command-light"


async def test_basic_llh(client):
    resp = await client.loglikelihood(model=TEST_MODEL, prompt="co:here", completion="co:where?")
    assert isinstance(resp, LogLikelihoods)
    assert isinstance(resp.prompt_tokens[0].encoded, int)
