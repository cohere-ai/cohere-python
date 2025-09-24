#!/usr/bin/env python3
"""
Demo script for the configurable batch size feature in Cohere SDK.

This demonstrates how to use the new batch_size and max_workers parameters
to control embedding batch processing.
"""

import os
import time
import cohere

# Initialize client (requires CO_API_KEY environment variable)
client = cohere.Client()

# Sample texts for embedding
texts = [f"Text document number {i}" for i in range(20)]

print(f"Embedding {len(texts)} texts...")
print()

# Example 1: Default behavior (batch_size=96)
print("1. Default behavior (batch_size=96):")
start = time.time()
response = client.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document"
)
print(f"   Time: {time.time() - start:.2f}s")
print(f"   Number of embeddings: {len(response.embeddings)}")
print()

# Example 2: Custom small batch size
print("2. Custom small batch size (batch_size=5):")
start = time.time()
response = client.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document",
    batch_size=5  # Will make 4 API calls for 20 texts
)
print(f"   Time: {time.time() - start:.2f}s")
print(f"   Number of embeddings: {len(response.embeddings)}")
print()

# Example 3: Custom batch size with fewer workers
print("3. Custom batch size with fewer workers (batch_size=5, max_workers=2):")
start = time.time()
response = client.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document",
    batch_size=5,
    max_workers=2  # Limit concurrency to 2 threads
)
print(f"   Time: {time.time() - start:.2f}s")
print(f"   Number of embeddings: {len(response.embeddings)}")
print()

# Example 4: Large batch size (all in one API call)
print("4. Large batch size (batch_size=100):")
start = time.time()
response = client.embed(
    texts=texts,
    model="embed-english-v3.0",
    input_type="search_document",
    batch_size=100  # All texts in a single API call
)
print(f"   Time: {time.time() - start:.2f}s")
print(f"   Number of embeddings: {len(response.embeddings)}")
print()

print("Demo completed!")
print()
print("Key benefits of configurable batch size:")
print("- batch_size: Control memory usage and API call granularity")
print("- max_workers: Control concurrency for rate limiting or resource constraints")
print("- Backward compatible: Defaults to existing behavior if not specified")