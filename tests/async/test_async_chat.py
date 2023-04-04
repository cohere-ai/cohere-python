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


@pytest.mark.asyncio
async def test_async_chat_stream(async_client):
    res = await async_client.chat(
        query="wagmi",
        max_tokens=5,
        stream=True,
    )

    first_element = True
    async for token in res:
        if first_element:
            assert token.id != None
            assert token.session_id != None
            first_element = False
        else:
            assert token.id == None
            assert token.session_id == None

        assert isinstance(token.text, str)
        assert len(token.text) > 0

    assert isinstance(res.texts, list)


@pytest.mark.asyncio
async def test_async_id(async_client):
    res1 = await async_client.chat(
        query="wagmi",
        max_tokens=5,
    )
    assert isinstance(res1.id, str)

    res2 = await async_client.chat(
        query="wagmi",
        max_tokens=5,
    )
    assert isinstance(res2.id, str)

    assert res1.id != res2.id
