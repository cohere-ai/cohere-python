import pytest

from cohere.responses.classify import Example
from cohere.error import CohereError


@pytest.mark.asyncio
async def test_async_classify(async_client):
    prediction = await async_client.classify(
        model="small",
        inputs=["purple"],
        examples=[
            Example("apple", "fruit"),
            Example("banana", "fruit"),
            Example("cherry", "fruit"),
            Example("watermelon", "fruit"),
            Example("kiwi", "fruit"),
            Example("red", "color"),
            Example("blue", "color"),
            Example("green", "color"),
            Example("yellow", "color"),
            Example("magenta", "color"),
        ],
    )
    assert isinstance(prediction.classifications, list)
    assert prediction.meta
    assert prediction.meta["api_version"]
    assert prediction.meta["api_version"]["version"]


@pytest.mark.asyncio
async def test_async_classify_input_validation(async_client):
    for value in [None, []]:
        with pytest.raises(CohereError) as exc:
            await async_client.classify(
                model="small",
                inputs=value,
                examples=[
                    Example("apple", "fruit"),
                    Example("kiwi", "fruit"),
                    Example("yellow", "color"),
                    Example("magenta", "color"),
                ],
            )
        assert "inputs must be a non-empty list of strings." in str(exc.value)

    for value in [None, []]:
        with pytest.raises(CohereError) as exc:
            await async_client.classify(
                model="small",
                inputs=["apple", "yellow"],
                examples=value,
            )
        assert "examples must be a non-empty list of ClassifyExample objects." in str(exc.value)
