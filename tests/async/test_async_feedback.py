import pytest

from cohere.responses.feedback import GenerateFeedbackResponse


@pytest.mark.asyncio
async def test_async_generate_feedback_basic(async_client):
    generations = await async_client.generate(prompt="co:here", max_tokens=1)
    feedback = await async_client.generate_feedback(request_id=generations[0].id,
                                                    desired_response="is the best",
                                                    good_response=False)
    assert isinstance(feedback, GenerateFeedbackResponse)
