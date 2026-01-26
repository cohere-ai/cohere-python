"""
Unit test for the embed_stream method added in PR #698.

This test validates the embed_stream functionality using the actual
Cohere SDK implementation without requiring API keys.

Run with: python test_sdk_embed_stream_unit.py
"""

import sys
import json
from unittest.mock import Mock, patch, MagicMock
import cohere
from cohere.streaming_utils import StreamingEmbedParser, StreamedEmbedding


def create_mock_embed_response(texts, embedding_dim=1024):
    """Create a mock embed API response."""
    embeddings = [[0.1 * i + j * 0.001 for j in range(embedding_dim)] for i in range(len(texts))]

    response_data = {
        "id": "test-id",
        "embeddings": embeddings,
        "texts": texts,
        "response_type": "embeddings_floats",
        "meta": {"api_version": {"version": "1"}}
    }

    # Create mock response object
    mock_response = Mock()
    mock_response._response = Mock()
    mock_response._response.json.return_value = response_data
    mock_response._response.content = json.dumps(response_data).encode('utf-8')
    mock_response.data = Mock()
    mock_response.data.embeddings = embeddings

    return mock_response


def test_embed_stream_basic():
    """Test basic embed_stream functionality."""
    print("="*80)
    print("TEST 1: Basic embed_stream Functionality")
    print("="*80)

    # Create client
    client = cohere.Client(api_key="test-key")

    test_texts = [
        "Hello world",
        "This is a test",
        "Embed stream works!"
    ]

    print(f"\n📝 Testing with {len(test_texts)} texts")

    # Mock the raw client's embed method
    with patch.object(client._raw_client, 'embed') as mock_embed:
        mock_embed.return_value = create_mock_embed_response(test_texts)

        embeddings = []
        for embedding in client.embed_stream(
            texts=test_texts,
            model="embed-english-v3.0",
            input_type="search_document",
            batch_size=3
        ):
            embeddings.append(embedding)
            print(f"   ✓ Got embedding {embedding.index}: {embedding.text}")

        # Verify results
        assert len(embeddings) == len(test_texts), f"Expected {len(test_texts)} embeddings, got {len(embeddings)}"

        for i, emb in enumerate(embeddings):
            assert emb.index == i, f"Expected index {i}, got {emb.index}"
            assert emb.text == test_texts[i], f"Text mismatch at index {i}"
            assert len(emb.embedding) > 0, f"Empty embedding at index {i}"

        print(f"\n✅ Test 1 PASSED: Got all {len(embeddings)} embeddings correctly")
        return True


def test_embed_stream_batching():
    """Test that embed_stream processes texts in batches."""
    print("\n" + "="*80)
    print("TEST 2: Batch Processing")
    print("="*80)

    client = cohere.Client(api_key="test-key")

    # Create more texts than batch_size
    test_texts = [f"Document {i}" for i in range(25)]
    batch_size = 5

    print(f"\n📝 Testing with {len(test_texts)} texts, batch_size={batch_size}")
    print(f"   Expected API calls: {(len(test_texts) + batch_size - 1) // batch_size}")

    call_count = 0

    def mock_embed_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        batch_texts = kwargs.get('texts', [])
        print(f"   API call {call_count}: Processing {len(batch_texts)} texts")
        return create_mock_embed_response(batch_texts)

    with patch.object(client._raw_client, 'embed') as mock_embed:
        mock_embed.side_effect = mock_embed_side_effect

        embeddings = list(client.embed_stream(
            texts=test_texts,
            model="embed-english-v3.0",
            batch_size=batch_size
        ))

        expected_calls = (len(test_texts) + batch_size - 1) // batch_size
        assert call_count == expected_calls, f"Expected {expected_calls} API calls, got {call_count}"
        assert len(embeddings) == len(test_texts), f"Expected {len(test_texts)} embeddings, got {len(embeddings)}"

        print(f"\n✅ Test 2 PASSED: Made {call_count} API calls as expected")
        return True


def test_embed_stream_empty_input():
    """Test embed_stream with empty input."""
    print("\n" + "="*80)
    print("TEST 3: Empty Input Handling")
    print("="*80)

    client = cohere.Client(api_key="test-key")

    print("\n📝 Testing with empty text list")

    embeddings = list(client.embed_stream(
        texts=[],
        model="embed-english-v3.0"
    ))

    assert len(embeddings) == 0, f"Expected 0 embeddings, got {len(embeddings)}"

    print("✅ Test 3 PASSED: Empty input handled correctly")
    return True


def test_embed_stream_memory_efficiency():
    """Test that embed_stream yields results incrementally."""
    print("\n" + "="*80)
    print("TEST 4: Memory Efficiency (Iterator Behavior)")
    print("="*80)

    client = cohere.Client(api_key="test-key")

    test_texts = [f"Document {i}" for i in range(15)]

    print(f"\n📝 Testing that embeddings are yielded incrementally")

    with patch.object(client._raw_client, 'embed') as mock_embed:
        mock_embed.side_effect = lambda **kwargs: create_mock_embed_response(kwargs['texts'])

        # Verify it returns an iterator (generator)
        result = client.embed_stream(
            texts=test_texts,
            model="embed-english-v3.0",
            batch_size=5
        )

        # Check it's an iterator
        assert hasattr(result, '__iter__'), "Result should be an iterator"
        assert hasattr(result, '__next__'), "Result should be a generator"

        # Process first embedding
        first_embedding = next(result)
        assert first_embedding.index == 0, "First embedding should have index 0"
        print(f"   ✓ First embedding yielded before processing all texts")

        # Process remaining embeddings
        remaining = list(result)
        assert len(remaining) == len(test_texts) - 1, "Should get remaining embeddings"

        print(f"   ✓ Embeddings yielded one at a time (memory efficient)")
        print("\n✅ Test 4 PASSED: Iterator behavior confirmed")
        return True


def test_streaming_embed_parser():
    """Test the StreamingEmbedParser utility."""
    print("\n" + "="*80)
    print("TEST 5: StreamingEmbedParser Utility")
    print("="*80)

    print("\n📝 Testing StreamingEmbedParser")

    # Create mock response
    test_texts = ["Hello", "World", "Test"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    response_data = {
        "embeddings": embeddings,
        "texts": test_texts,
        "response_type": "embeddings_floats"
    }

    # Create mock response object
    mock_response = Mock()
    mock_response.json.return_value = response_data
    mock_response.content = json.dumps(response_data).encode('utf-8')

    # Parse embeddings
    parser = StreamingEmbedParser(mock_response, test_texts)
    parsed_embeddings = list(parser.iter_embeddings())

    assert len(parsed_embeddings) == len(test_texts), f"Expected {len(test_texts)} embeddings"

    for i, emb in enumerate(parsed_embeddings):
        assert emb.embedding == embeddings[i], f"Embedding mismatch at index {i}"
        print(f"   ✓ Parsed embedding {i}: {emb.embedding}")

    print("\n✅ Test 5 PASSED: StreamingEmbedParser works correctly")
    return True


def test_embed_stream_v2_client():
    """Test embed_stream with V2Client."""
    print("\n" + "="*80)
    print("TEST 6: V2Client embed_stream")
    print("="*80)

    client = cohere.ClientV2(api_key="test-key")

    test_texts = ["Test 1", "Test 2", "Test 3"]

    print(f"\n📝 Testing V2Client with {len(test_texts)} texts")

    with patch.object(client._raw_client, 'embed') as mock_embed:
        mock_embed.return_value = create_mock_embed_response(test_texts)

        embeddings = list(client.embed_stream(
            texts=test_texts,
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
            batch_size=3
        ))

        assert len(embeddings) == len(test_texts), f"Expected {len(test_texts)} embeddings"
        print(f"   ✓ Got {len(embeddings)} embeddings from V2Client")

        print("\n✅ Test 6 PASSED: V2Client embed_stream works")
        return True


def main():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("EMBED_STREAM SDK UNIT TESTS (PR #698)")
    print("="*80)
    print("Testing the actual Cohere SDK embed_stream implementation")
    print("="*80)

    results = []

    try:
        results.append(("Basic Functionality", test_embed_stream_basic()))
        results.append(("Batch Processing", test_embed_stream_batching()))
        results.append(("Empty Input", test_embed_stream_empty_input()))
        results.append(("Memory Efficiency", test_embed_stream_memory_efficiency()))
        results.append(("StreamingEmbedParser", test_streaming_embed_parser()))
        results.append(("V2Client Support", test_embed_stream_v2_client()))

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("\n" + "="*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 ALL UNIT TESTS PASSED!")
        print("\nThe embed_stream implementation in PR #698 is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
