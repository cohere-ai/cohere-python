import pytest

from cohere.responses.feedback import (
    GenerateFeedbackResponse,
    GeneratePreferenceFeedbackResponse,
)


@pytest.mark.asyncio
async def test_async_generate_feedback_basic(async_client):
    generations = await async_client.generate(prompt="co:here", max_tokens=1)
    feedback = await async_client.generate_feedback(request_id=generations[0].id,
                                                    desired_response="is the best",
                                                    good_response=False)
    assert isinstance(feedback, GenerateFeedbackResponse)


@pytest.mark.asyncio
async def test_async_generate_preference_feedback_basic(async_client):
    generations = await async_client.generate(prompt="co:here", max_tokens=1, num_generations=2)
    feedback = await async_client.generate_preference_feedback(prompt="co:here",
                                                               ratings=[
                                                                   (generations[0].id, 0.5, generations[0].generation),
                                                                   (generations[1].id, 0.5, generations[1].generation)
                                                               ])
    assert isinstance(feedback, GeneratePreferenceFeedbackResponse)
