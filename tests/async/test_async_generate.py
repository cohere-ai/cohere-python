import pytest

from cohere import CohereAPIError
from cohere.responses import Generations

TEST_MODEL = "base-light"


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
    # predictions = await async_client.generate(model=TEST_MODEL, prompt="Hello,", max_tokens=3, num_generations=5)
    # assert sorted(predictions, key=lambda p: -p.likelihood) == predictions  # they should be in order of likelihood


@pytest.mark.asyncio
async def test_raise_ex(async_client):
    with pytest.raises(CohereAPIError):
        await async_client.generate(prompt="too long", max_tokens=100000)

    with pytest.raises(CohereAPIError):
        await async_client.batch_generate(
            ["not too long", "way too long even if we support 8 k tokens" * 2000],
            max_tokens=10,
            return_exceptions=False,
        )

    multi = await async_client.batch_generate(
        ["not too long", "way too long even if we support 8 k tokens" * 2000], max_tokens=10, return_exceptions=True
    )
    assert isinstance(multi[0], Generations)
    assert isinstance(multi[1], CohereAPIError)


@pytest.mark.asyncio
async def test_async_generate_stream(async_client):
    res = await async_client.generate(
        "Hey!", max_tokens=5, stream=True, temperature=0
    )  # setting temp=0 to avoid random finish reasons
    final_text = ""
    async for token in res:
        assert isinstance(token.text, str)
        assert len(token.text) > 0
        assert token.index == 0
        assert not token.is_finished
        final_text += token.text

    assert res.id != None
    assert res.finish_reason, "MAX_TOKENS"

    assert isinstance(res.generations, Generations)
    assert res.generations[0].finish_reason == "MAX_TOKENS"
    assert res.generations[0].prompt == "Hey!"
    assert res.generations[0].text == final_text
    assert res.generations[0].id != None

    assert isinstance(res.texts, list)
    assert len(res.texts) > 0
