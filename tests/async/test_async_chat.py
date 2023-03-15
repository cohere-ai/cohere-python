import pytest


@pytest.mark.asyncio
async def test_async_multi_replies(async_client):
    num_replies = 3
    prediction = await async_client.chat("Yo what's up?", return_chatlog=True)
    assert prediction.chatlog is not None
    for _ in range(num_replies):
        prediction = await prediction.respond("oh that's cool")
        assert isinstance(prediction.reply, str)
        assert isinstance(prediction.session_id, str)
        assert prediction.persona_name is None
        assert prediction.chatlog is not None
        assert prediction.meta
        assert prediction.meta["api_version"]
        assert prediction.meta["api_version"]["version"]
