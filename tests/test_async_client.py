"""
Test script for async client migration from httpx to aiohttp.

This script tests various async operations to ensure the aiohttp integration works correctly.
Run with: python test_async_client.py
"""

import asyncio
import os
from cohere.client import AsyncClient
from cohere.client_v2 import AsyncClientV2

# Set your API key here or use environment variable
API_KEY = os.getenv("CO_API_KEY", "your-api-key-here")


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


if __name__ == "__main__":
    asyncio.run(main())
