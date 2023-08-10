from utils import get_api_key

import cohere
from cohere.responses import LogLikelihoods

TEST_MODEL = "command-light"
API_KEY = get_api_key()
co = cohere.Client(API_KEY)


async def test_basic_llh():
    resp = await co.loglikelihood(model=TEST_MODEL, prompt="co:here", completion="co:where?")
    assert isinstance(resp, LogLikelihoods)
    assert isinstance(resp.prompt_tokens[0].encoded, int)
