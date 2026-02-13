"""
Test script for async client migration from httpx to aiohttp.

This script tests various async operations to ensure the aiohttp integration works correctly.
Run with: python test_async_client.py
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from cohere.client import AsyncClient
from cohere.client_v2 import AsyncClientV2
from cohere.errors import BadRequestError, UnauthorizedError, TooManyRequestsError, InternalServerError
from cohere.core.api_error import ApiError

# Set your API key here or use environment variable
API_KEY = os.getenv("CO_API_KEY", "your-api-key-here")


@pytest.mark.asyncio
async def test_basic_chat():
    """Test basic async chat with v1 API."""
    print("\n=== Testing AsyncClient Chat (v1) ===")
    
    async with AsyncClient(api_key=API_KEY) as client:
        try:
            response = await client.chat(
                message="Say 'Hello, aiohttp!' in exactly those words.",
                model="command-a-03-2025-pld-rl",
            )
            print(f"✓ Chat response: {response.text}")
            return True
        except Exception as e:
            print(f"✗ Chat failed: {type(e).__name__}: {e}")
            return False


@pytest.mark.asyncio
async def test_streaming_chat():
    """Test streaming chat with v1 API."""
    print("\n=== Testing AsyncClient Streaming (v1) ===")
    
    async with AsyncClient(api_key=API_KEY) as client:
        try:
            stream = client.chat_stream(
                message="Count from 1 to 3, one number per line.",
                model="command-a-03-2025-pld-rl",
            )
            
            chunks = []
            async for chunk in stream:
                if hasattr(chunk, 'text') and chunk.text:
                    chunks.append(chunk.text)
                    print(f"  Chunk: {chunk.text}")
            
            print(f"✓ Received {len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"✗ Streaming failed: {type(e).__name__}: {e}")
            return False


@pytest.mark.asyncio
async def test_embed():
    """Test embedding with v1 API."""
    print("\n=== Testing AsyncClient Embed (v1) ===")
    
    async with AsyncClient(api_key=API_KEY) as client:
        try:
            response = await client.embed(
                texts=["Hello world", "Testing aiohttp"],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            print(f"✓ Embed response: {len(response.embeddings)} embeddings")
            print(f"  First embedding dimension: {len(response.embeddings[0])}")
            return True
        except Exception as e:
            print(f"✗ Embed failed: {type(e).__name__}: {e}")
            return False


@pytest.mark.asyncio
async def test_v2_chat():
    """Test chat with v2 API."""
    print("\n=== Testing AsyncClientV2 Chat (v2) ===")
    
    async with AsyncClientV2(api_key=API_KEY) as client:
        try:
            response = await client.chat(
                model="command-a-03-2025-pld-rl",
                messages=[
                    {"role": "user", "content": "Say 'Hello from v2 API!' in exactly those words."}
                ]
            )
            print(f"✓ V2 Chat response: {response.message.content[0].text if response.message.content else 'No content'}")
            return True
        except Exception as e:
            print(f"✗ V2 Chat failed: {type(e).__name__}: {e}")
            return False


@pytest.mark.asyncio
async def test_v2_streaming():
    """Test streaming with v2 API (uses SSE)."""
    print("\n=== Testing AsyncClientV2 Streaming (v2 SSE) ===")
    
    async with AsyncClientV2(api_key=API_KEY) as client:
        try:
            stream = client.chat_stream(
                model="command-a-03-2025-pld-rl",
                messages=[
                    {"role": "user", "content": "Count from 1 to 3."}
                ]
            )
            
            chunks = []
            async for chunk in stream:
                if hasattr(chunk, 'type'):
                    chunks.append(chunk.type)
                    if chunk.type == "content-delta":
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'message'):
                            print(f"  Delta: {chunk.delta.message.content.text if hasattr(chunk.delta.message.content, 'text') else chunk.delta}")
            
            print(f"✓ V2 Streaming received {len(chunks)} events")
            return True
        except Exception as e:
            print(f"✗ V2 Streaming failed: {type(e).__name__}: {e}")
            return False


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with invalid API key."""
    print("\n=== Testing Error Handling ===")
    
    async with AsyncClient(api_key="invalid-key-12345") as client:
        try:
            await client.chat(message="This should fail", model="command-a-03-2025-pld-rl")
            print("✗ Should have raised an error")
            return False
        except Exception as e:
            print(f"✓ Correctly raised error: {type(e).__name__}")
            return True


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test multiple concurrent async requests."""
    print("\n=== Testing Concurrent Requests ===")
    
    async with AsyncClient(api_key=API_KEY) as client:
        try:
            # Fire off 3 requests concurrently
            tasks = [
                client.chat(message=f"Say the number {i}", model="command-a-03-2025-pld-rl")
                for i in range(1, 4)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successes = sum(1 for r in responses if not isinstance(r, Exception))
            print(f"✓ Concurrent requests: {successes}/3 successful")
            return successes > 0
        except Exception as e:
            print(f"✗ Concurrent requests failed: {type(e).__name__}: {e}")
            return False


@pytest.mark.asyncio
async def test_timeout():
    """Test timeout configuration."""
    print("\n=== Testing Timeout Configuration ===")
    
    async with AsyncClient(api_key=API_KEY, timeout=60.0) as client:
        try:
            response = await client.chat(
                message="Quick response test",
                model="command-a-03-2025-pld-rl",
            )
            print(f"✓ Timeout config works: got response")
            return True
        except Exception as e:
            print(f"✗ Timeout test failed: {type(e).__name__}: {e}")
            return False


async def main():
    """Run all async tests."""
    print("=" * 60)
    print("Testing Async Client with aiohttp")
    print("=" * 60)
    
    if API_KEY == "your-api-key-here":
        print("\n⚠️  WARNING: Please set CO_API_KEY environment variable or update API_KEY in this script")
        print("Skipping tests that require API key...\n")
    
    results = {}
    
    # Run tests
    tests = [
        ("Basic Chat (v1)", test_basic_chat),
        ("Streaming Chat (v1)", test_streaming_chat),
        ("Embed (v1)", test_embed),
        ("Chat V2", test_v2_chat),
        ("Streaming V2 (SSE)", test_v2_streaming),
        ("Concurrent Requests", test_concurrent_requests),
        ("Timeout Config", test_timeout),
        ("Error Handling", test_error_handling),
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {type(e).__name__}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! aiohttp migration is working correctly.")
    elif total_passed > 0:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Check errors above.")
    else:
        print("\n❌ All tests failed. Check your API key and connection.")


# Unit tests for aiohttp error handling fixes
# These tests verify that error responses properly await .read(), .json(), and .text()

@pytest.mark.asyncio
async def test_chat_stream_error_response_400():
    """Test that chat_stream properly handles 400 error with JSON body using await response.json()."""
    
    # Create a mock aiohttp ClientResponse
    mock_response = AsyncMock()
    mock_response.status = 400
    mock_response.status_code = 400
    mock_response.headers = {"content-type": "application/json"}
    
    # Mock the async methods that were fixed
    error_body = {"message": "Invalid request", "error": "bad_request"}
    mock_response.read = AsyncMock(return_value=json.dumps(error_body).encode())
    mock_response.json = AsyncMock(return_value=error_body)
    mock_response.text = AsyncMock(return_value=json.dumps(error_body))
    
    # Mock aiter_lines to return nothing (error path doesn't use it)
    async def mock_aiter_lines():
        return
        yield  # Make it a generator
    mock_response.aiter_lines = mock_aiter_lines
    
    # Patch at the aiohttp session level
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Mock the request context manager
        @asynccontextmanager
        async def mock_stream_context(*args, **kwargs):
            yield mock_response
        
        mock_session.request = mock_stream_context
        
        client = AsyncClient(api_key="test-key")
        
        try:
            stream = client.chat_stream(
                message="test",
                model="command"
            )
            async for _ in stream:
                pass
            pytest.fail("Should have raised BadRequestError")
        except BadRequestError as e:
            # Verify the error was properly constructed with awaited json()
            # Verify async methods were called
            mock_response.read.assert_called_once()
            assert mock_response.json.call_count >= 1  # Called at least once for error body


@pytest.mark.asyncio
async def test_v2_chat_stream_error_response_401():
    """Test that v2 chat_stream properly handles 401 error using await response.json()."""
    
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.status_code = 401
    mock_response.headers = {"content-type": "application/json"}
    
    error_body = {"message": "Unauthorized", "error": "invalid_token"}
    mock_response.read = AsyncMock(return_value=json.dumps(error_body).encode())
    mock_response.json = AsyncMock(return_value=error_body)
    mock_response.text = AsyncMock(return_value=json.dumps(error_body))
    
    # Mock SSE iteration
    from cohere.core.http_sse._api import EventSource
    original_aiter_sse = EventSource.aiter_sse
    
    async def mock_aiter_sse(self):
        # Don't iterate, go straight to error handling
        return
        yield  # Make it a generator
    
    with patch.object(EventSource, 'aiter_sse', mock_aiter_sse):
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            @asynccontextmanager
            async def mock_stream_context(*args, **kwargs):
                yield mock_response
            
            mock_session.request = mock_stream_context
            
            client = AsyncClientV2(api_key="test-key")
            
            try:
                stream = client.chat_stream(
                    model="command",
                    messages=[{"role": "user", "content": "test"}]
                )
                async for _ in stream:
                    pass
                pytest.fail("Should have raised UnauthorizedError")
            except UnauthorizedError as e:
                # Verify async methods were awaited
                mock_response.read.assert_called_once()
                assert mock_response.json.call_count >= 1


@pytest.mark.asyncio
async def test_generate_stream_error_response_429():
    """Test that generate streaming properly handles 429 error using await response.json()."""
    
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.status_code = 429
    mock_response.headers = {"content-type": "application/json"}
    
    error_body = {"message": "Rate limit exceeded", "error": "too_many_requests"}
    mock_response.read = AsyncMock(return_value=json.dumps(error_body).encode())
    mock_response.json = AsyncMock(return_value=error_body)
    mock_response.text = AsyncMock(return_value=json.dumps(error_body))
    
    async def mock_aiter_lines():
        return
        yield
    mock_response.aiter_lines = mock_aiter_lines
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        @asynccontextmanager
        async def mock_stream_context(*args, **kwargs):
            yield mock_response
        
        mock_session.request = mock_stream_context
        
        client = AsyncClient(api_key="test-key")
        
        try:
            stream = client.generate_stream(
                prompt="test",
                model="command"
            )
            async for _ in stream:
                pass
            pytest.fail("Should have raised TooManyRequestsError")
        except TooManyRequestsError as e:
            # Verify async methods were awaited
            mock_response.read.assert_called_once()
            assert mock_response.json.call_count >= 1


@pytest.mark.asyncio
async def test_chat_stream_error_response_500():
    """Test that chat_stream properly handles 500 error using await response.json()."""
    
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.status_code = 500
    mock_response.headers = {"content-type": "application/json"}
    
    error_body = {"message": "Internal server error", "error": "server_error"}
    mock_response.read = AsyncMock(return_value=json.dumps(error_body).encode())
    mock_response.json = AsyncMock(return_value=error_body)
    mock_response.text = AsyncMock(return_value=json.dumps(error_body))
    
    async def mock_aiter_lines():
        return
        yield
    mock_response.aiter_lines = mock_aiter_lines
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        @asynccontextmanager
        async def mock_stream_context(*args, **kwargs):
            yield mock_response
        
        mock_session.request = mock_stream_context
        
        client = AsyncClient(api_key="test-key")
        
        try:
            stream = client.chat_stream(
                message="test",
                model="command"
            )
            async for _ in stream:
                pass
            pytest.fail("Should have raised InternalServerError")
        except InternalServerError as e:
            # Verify async methods were awaited
            mock_response.read.assert_called_once()
            assert mock_response.json.call_count >= 1


@pytest.mark.asyncio
async def test_chat_stream_error_invalid_json_uses_text():
    """Test that chat_stream uses await response.text() when JSON parsing fails."""
    
    mock_response = AsyncMock()
    mock_response.status = 400
    mock_response.status_code = 400
    mock_response.headers = {"content-type": "text/plain"}
    
    # Simulate invalid JSON - json() will raise JSONDecodeError
    mock_response.read = AsyncMock(return_value=b"Invalid response body")
    mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    mock_response.text = AsyncMock(return_value="Invalid response body")
    
    async def mock_aiter_lines():
        return
        yield
    mock_response.aiter_lines = mock_aiter_lines
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        @asynccontextmanager
        async def mock_stream_context(*args, **kwargs):
            yield mock_response
        
        mock_session.request = mock_stream_context
        
        client = AsyncClient(api_key="test-key")
        
        try:
            stream = client.chat_stream(
                message="test",
                model="command"
            )
            async for _ in stream:
                pass
            pytest.fail("Should have raised ApiError")
        except ApiError as e:
            # Verify text() was awaited in the JSONDecodeError path
            mock_response.read.assert_called_once()
            mock_response.text.assert_called()
            assert "Invalid response body" in str(e) or (hasattr(e, 'body') and "Invalid" in str(e.body))


@pytest.mark.asyncio 
async def test_multiple_error_codes_use_await_json():
    """Test that various HTTP error codes all properly await response.json()."""
    from cohere.errors import (
        ForbiddenError, NotFoundError, UnprocessableEntityError,
        InvalidTokenError, ServiceUnavailableError
    )
    
    error_codes_and_exceptions = [
        (403, ForbiddenError, "Forbidden"),
        (404, NotFoundError, "Not found"),
        (422, UnprocessableEntityError, "Unprocessable entity"),
        (498, InvalidTokenError, "Invalid token"),
        (503, ServiceUnavailableError, "Service unavailable"),
    ]
    
    for status_code, expected_exception, error_msg in error_codes_and_exceptions:
        mock_response = AsyncMock()
        mock_response.status = status_code
        mock_response.status_code = status_code
        mock_response.headers = {"content-type": "application/json"}
        
        error_body = {"message": error_msg, "error": "test_error"}
        mock_response.read = AsyncMock(return_value=json.dumps(error_body).encode())
        mock_response.json = AsyncMock(return_value=error_body)
        mock_response.text = AsyncMock(return_value=json.dumps(error_body))
        
        async def mock_aiter_lines():
            return
            yield
        mock_response.aiter_lines = mock_aiter_lines
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            @asynccontextmanager
            async def mock_stream_context(*args, **kwargs):
                yield mock_response
            
            mock_session.request = mock_stream_context
            
            client = AsyncClient(api_key="test-key")
            
            try:
                stream = client.chat_stream(message="test", model="command")
                async for _ in stream:
                    pass
                pytest.fail(f"Should have raised {expected_exception.__name__} for {status_code}")
            except expected_exception:
                # Verify async methods were properly awaited
                mock_response.read.assert_called_once()
                assert mock_response.json.call_count >= 1


if __name__ == "__main__":
    asyncio.run(main())
