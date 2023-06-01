import pytest


@pytest.mark.asyncio
async def test_async_multi_replies(async_client):
    num_replies = 3
    prediction = await async_client.chat("Yo what's up?", return_chatlog=True, max_tokens=5)
    assert prediction.chatlog is not None
    for _ in range(num_replies):
        prediction = await prediction.respond("oh that's cool", max_tokens=5)
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

    assert res is not None
    assert isinstance(res.texts, list)
    assert len(res.texts) == 0
    assert res.conversation_id is None
    assert res.response_id is None

    expected_index = 0
    expected_text = ""
    async for token in res:
        assert isinstance(token.text, str)
        assert len(token.text) > 0
        assert token.index == expected_index

        expected_index += 1
        expected_text += token.text

    assert res.texts == [expected_text]
    assert res.conversation_id is not None
    assert res.response_id is not None


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
