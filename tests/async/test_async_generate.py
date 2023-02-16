import pytest

from cohere import CohereAPIError
from cohere.responses import BatchedGenerations, Generation, Generations

TEST_MODEL = "medium"
VERBOTEN = "Nazi Germany was the German state between 1933 and 1945, when Adolf Hitler controlled the country, transforming it into a dictatorship."  # blocked sentence from wikipedia


@pytest.mark.asyncio
async def test_single_generate(client):
    prediction = await client.generate(model=TEST_MODEL, prompt="co:here", max_tokens=1)
    assert isinstance(prediction, Generations)
    assert isinstance(prediction.text, str)
    assert str(prediction[0]) == prediction.text
    #assert str(prediction) == prediction.text
    assert prediction[0][:5] == prediction.text[:5]
    assert prediction[0].likelihood is None
    assert prediction.prompt == "co:here"
    assert prediction[0].prompt == "co:here"
    # test repr
    assert prediction[0].visualize_token_likelihoods() is None
    assert isinstance(prediction[0]._repr_html_(), str)  # Generation
    assert isinstance(prediction._repr_html_(), str)  # Generations repr


@pytest.mark.asyncio
async def test_batch_generate(client):
    predictions = await client.generate(model=TEST_MODEL, prompt=["co:here"] * 2, max_tokens=1)
    assert isinstance(predictions[0], Generation)


@pytest.mark.asyncio
async def test_batch_generate(client):
    predictions = await client.generate(model=TEST_MODEL, prompt=["co:here"] * 2, num_generations=2, max_tokens=1)
    assert isinstance(predictions, BatchedGenerations)
    assert isinstance(predictions[0], Generations)
    assert isinstance(predictions[0][0][:5], str)
    assert isinstance(predictions._repr_html_(), str)  # BatchGenerations repr


@pytest.mark.asyncio
async def test_return_likelihoods_generation(client):
    prediction = await client.generate(
        model=TEST_MODEL, prompt="co:here", max_tokens=1, return_likelihoods="GENERATION"
    )
    assert prediction[0].token_likelihoods[0].token
    assert "<span" in prediction[0].visualize_token_likelihoods(display=False)
    prediction = await client.generate(model=TEST_MODEL, prompt="co:here", max_tokens=1, return_likelihoods=True)
    assert prediction[0].token_likelihoods[0].token
    predictions = await client.generate(model=TEST_MODEL, prompt="Hello,", max_tokens=3, num_generations=5)
    #assert sorted(predictions, key=lambda p: -p.likelihood) == predictions  # they should be in order of likelihood


@pytest.mark.asyncio
async def test_command(client):
    prediction = await client.command("Give me a recipe for pizza.", max_tokens=25)
    assert isinstance(prediction, Generations)
    assert isinstance(prediction.text, str)


@pytest.mark.asyncio
async def test_raise_ex(client):
    with pytest.raises(CohereAPIError):
        await client.generate(prompt=VERBOTEN, max_tokens=1, return_likelihoods="GENERATION")
    with pytest.raises(CohereAPIError):
        await client.generate(prompt=["innocent", VERBOTEN], max_tokens=1, return_likelihoods="GENERATION")


@pytest.mark.asyncio
async def test_return_ex(client_retex):
    response = await client_retex.generate(prompt=VERBOTEN, max_tokens=1, return_likelihoods="GENERATION")
    assert isinstance(response, CohereAPIError)

    multi = await client_retex.generate(prompt=["innocent", VERBOTEN], max_tokens=1, return_likelihoods="GENERATION")
    assert isinstance(multi[0], Generations)
    assert isinstance(multi[1], CohereAPIError)

    assert isinstance(multi._repr_html_(), str)  # no crashes
