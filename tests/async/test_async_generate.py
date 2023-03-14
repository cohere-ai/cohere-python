import pytest

from cohere import CohereAPIError
from cohere.responses import Generation, Generations

TEST_MODEL = "medium"
VERBOTEN = "Nazi Germany was the German state between 1933 and 1945, when Adolf Hitler controlled the country, transforming it into a dictatorship."  # blocked sentence from wikipedia


@pytest.mark.asyncio
async def test_single_generate(async_client):
    prediction = await async_client.generate(model=TEST_MODEL, prompt="co:here", max_tokens=1)
    assert isinstance(prediction, Generations)
    assert isinstance(prediction[0].text, str)
    assert str(prediction[0]) == prediction[0].text
    assert prediction[0].likelihood is None
    assert prediction.prompt == "co:here"
    assert prediction[0].prompt == "co:here"
    # test repr
    assert prediction[0].visualize_token_likelihoods() is None
    assert prediction.meta
    assert prediction.meta["api_version"]
    assert prediction.meta["api_version"]["version"]

@pytest.mark.asyncio
async def test_return_likelihoods_generation(async_client):
    prediction = await async_client.generate(
        model=TEST_MODEL, prompt="co:here", max_tokens=1, return_likelihoods="GENERATION"
    )
    assert prediction[0].token_likelihoods[0].token
    # pandas is optional
    # assert "<span" in prediction[0].visualize_token_likelihoods(display=False)
    prediction = await async_client.generate(model=TEST_MODEL, prompt="co:here", max_tokens=1, return_likelihoods="ALL")
    assert prediction[0].token_likelihoods[0].token
    #predictions = await async_client.generate(model=TEST_MODEL, prompt="Hello,", max_tokens=3, num_generations=5)
    #assert sorted(predictions, key=lambda p: -p.likelihood) == predictions  # they should be in order of likelihood


@pytest.mark.asyncio
async def test_raise_ex(async_client):
    with pytest.raises(CohereAPIError):
        await async_client.generate(prompt=VERBOTEN, max_tokens=1, return_likelihoods="GENERATION")
    with pytest.raises(CohereAPIError):
        await async_client.batch_generate(["innocent", VERBOTEN], max_tokens=1, return_likelihoods="GENERATION",return_exceptions=False)

    
    multi = await async_client.batch_generate(["innocent", VERBOTEN], max_tokens=1, return_likelihoods="GENERATION",return_exceptions=True)
    assert isinstance(multi[0], Generations)
    assert isinstance(multi[1], CohereAPIError)
