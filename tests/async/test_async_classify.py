import pytest
from cohere.responses.classify import Example

@pytest.mark.asyncio
async def test_summarize(async_client):
    prediction = await async_client.classify(model='small',
                                inputs=['purple'],
                                examples=[
                                    Example('apple', 'fruit'),
                                    Example('banana', 'fruit'),
                                    Example('cherry', 'fruit'),
                                    Example('watermelon', 'fruit'),
                                    Example('kiwi', 'fruit'),
                                    Example('red', 'color'),
                                    Example('blue', 'color'),
                                    Example('green', 'color'),
                                    Example('yellow', 'color'),
                                    Example('magenta', 'color')
                                ])
    assert isinstance(prediction.classifications, list)


