import conftest
import pytest

import cohere
from cohere.responses.chat import (
    StreamCitationGeneration,
    StreamEnd,
    StreamQueryGeneration,
    StreamSearchResults,
    StreamStart,
    StreamTextGeneration,
)


@pytest.mark.asyncio
async def test_async_multi_replies(async_client):
    conversation_id = f"test_conv_{conftest.random_word()}"
    num_replies = 3
    prediction = await async_client.chat(
        "Yo what's up?", return_chatlog=True, max_tokens=5, conversation_id=conversation_id
    )
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
    conversation_id = f"test_conv_{conftest.random_word()}"
    res = await async_client.chat(
        message="wagmi",
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
    expected_text = ""
    async for token in res:
        if isinstance(token, cohere.responses.chat.StreamStart):
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

    assert res.texts == [expected_text]
    assert res.conversation_id is not None
    assert res.response_id is not None


@pytest.mark.asyncio
async def test_async_chat_with_connectors_stream(async_client):
    res = await async_client.chat(
        "How deep in the Mariana Trench",
        temperature=0,
        stream=True,
        connectors=[{"id": "web-search"}],
        prompt_truncation="AUTO",
    )

    assert isinstance(res, cohere.responses.chat.StreamingChat)
    assert isinstance(res.texts, list)
    assert len(res.texts) == 0
    assert res.response_id is None
    assert res.finish_reason is None

    expected_index = 0
    expected_text = ""

    count_stream_start = 0
    count_stream_end = 0
    count_text_generation = 0
    count_query_generation = 0
    count_citation_generation = 0
    count_search_results = 0
    async for token in res:
        if isinstance(token, StreamStart):
            count_stream_start += 1
            assert token.generation_id is not None
            assert not token.is_finished
            assert token.event_type == "stream-start"
        elif isinstance(token, StreamQueryGeneration):
            count_query_generation += 1
            assert token.search_queries is not None
            assert token.event_type == "search-queries-generation"
        elif isinstance(token, StreamSearchResults):
            count_search_results += 1
            assert token.documents is not None
            assert token.search_results is not None
            assert token.event_type == "search-results"
        elif isinstance(token, StreamCitationGeneration):
            count_citation_generation += 1
            assert token.citations is not None
            assert token.event_type == "citation-generation"
        elif isinstance(token, StreamTextGeneration):
            count_text_generation += 1
            assert isinstance(token.text, str)
            assert len(token.text) > 0
            expected_text += token.text
            assert not token.is_finished
            assert token.event_type == "text-generation"
        elif isinstance(token, StreamEnd):
            count_stream_end += 1
            assert token.finish_reason == "COMPLETE"
            assert token.is_finished
            assert token.event_type == "stream-end"
        assert isinstance(token.index, int)
        assert token.index == expected_index
        expected_index += 1

    assert count_stream_start == 1
    assert count_stream_end == 1
    assert count_search_results == 1
    assert count_citation_generation > 0
    assert count_query_generation > 0
    assert count_text_generation > 0

    assert res.texts == [expected_text]
    assert res.response_id is not None
    assert res.finish_reason is not None


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
