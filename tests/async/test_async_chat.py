import conftest
import pytest

import cohere


@pytest.mark.asyncio
async def test_async_multi_replies(async_client):
    conversation_id = f"test_conv_{conftest.random_word()}"
    num_replies = 3
    prediction = await async_client.chat(
        "Yo what's up?", return_chat_history=True, max_tokens=5, conversation_id=conversation_id
    )
    assert prediction.chat_history is not None
    for _ in range(num_replies):
        prediction = await prediction.respond("oh that's cool", max_tokens=5)
        assert isinstance(prediction.text, str)
        assert isinstance(prediction.conversation_id, str)
        assert prediction.chat_history is not None
        assert prediction.meta
        assert prediction.meta["api_version"]
        assert prediction.meta["api_version"]["version"]


@pytest.mark.asyncio
async def test_async_chat_stream(async_client):
    conversation_id = f"test_conv_{conftest.random_word()}"
    res = await async_client.chat(
        message="How deep in the Mariana Trench?",
        max_tokens=5,
        conversation_id=conversation_id,
        stream=True,
    )

    assert res is not None
    assert isinstance(res.texts, list)
    assert len(res.texts) == 0
    assert res.conversation_id is None
    assert res.response_id is None

    expected_index = 0
    saw_stream_start = False
    expected_text = ""
    async for token in res:
        if isinstance(token, cohere.responses.chat.StreamStart):
            saw_stream_start = True
            assert token.generation_id is not None
            assert not token.is_finished
        elif isinstance(token, cohere.responses.chat.StreamTextGeneration):
            assert isinstance(token.text, str)
            assert len(token.text) > 0
            expected_text += token.text
            assert not token.is_finished

        assert isinstance(token.index, int)
        assert token.index == expected_index
        expected_index += 1

    assert saw_stream_start, f"no stream start event conversation id is {conversation_id}"
    assert res.texts is not None, f"no text generated conversation id is {conversation_id}"
    assert res.texts == [
        expected_text
    ], f"final text generated is not the same as the combined text sent conversation id is {conversation_id}"
    assert res.conversation_id is not None
    assert res.response_id is not None


@pytest.mark.asyncio
async def test_async_id(async_client):
    res1 = await async_client.chat(
        message="wagmi",
        max_tokens=5,
    )
    assert isinstance(res1.response_id, str)

    res2 = await async_client.chat(
        message="wagmi",
        max_tokens=5,
    )
    assert isinstance(res2.response_id, str)

    assert res1.response_id != res2.response_id
