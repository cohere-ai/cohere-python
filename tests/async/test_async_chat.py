import pytest


@pytest.mark.asyncio
async def test_async_multi_replies(async_client):
    num_replies = 3
    prediction = await async_client.chat("Yo what's up?", return_chatlog=True)
    assert prediction.chatlog is not None
    for _ in range(num_replies):
        prediction = await async_client.chat(
            "oh that's cool", conversation_id=prediction.conversation_id, return_chatlog=True
        )
        assert isinstance(prediction.text, str)
        assert isinstance(prediction.conversation_id, str)
        assert prediction.chatlog is not None
        assert prediction.meta
        assert prediction.meta["api_version"]
        assert prediction.meta["api_version"]["version"]


@pytest.mark.asyncio
async def test_async_chat_stream(async_client):
    res = await async_client.chat(
        query="wagmi",
        max_tokens=5,
        stream=True,
    )

    async for token in res:
        assert isinstance(token.text, str)
        assert len(token.text) > 0

    assert isinstance(res.texts, list)


@pytest.mark.asyncio
async def test_async_id(async_client):
    res1 = await async_client.chat(
        query="wagmi",
        max_tokens=5,
    )
    assert isinstance(res1.response_id, str)

    res2 = await async_client.chat(
        query="wagmi",
        max_tokens=5,
    )
    assert isinstance(res2.response_id, str)

    assert res1.response_id != res2.response_id
