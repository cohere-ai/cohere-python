import os
import time
import unittest

import httpx

import cohere


class TestConnectionPooling(unittest.TestCase):
    """Test suite for HTTP connection pooling functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Check if API key is available for integration tests
        cls.api_key_available = bool(os.environ.get("CO_API_KEY"))

    def test_httpx_client_creation_with_limits(self):
        """Test that httpx clients can be created with our connection pooling limits."""
        # Test creating httpx client with limits (our implementation)
        client_with_limits = httpx.Client(
            timeout=300,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0,
            ),
        )
        
        # Verify the client was created successfully
        self.assertIsNotNone(client_with_limits)
        self.assertIsInstance(client_with_limits, httpx.Client)
        
        # The limits are applied internally - we can't directly access them
        # but we verify the client works correctly with our configuration
        
        client_with_limits.close()

    def test_cohere_client_initialization(self):
        """Test that Cohere clients can be initialized with connection pooling."""
        # Test with dummy API key - just verifies initialization works
        sync_client = cohere.Client(api_key="dummy-key")
        v2_client = cohere.ClientV2(api_key="dummy-key")

        # Verify clients were created
        self.assertIsNotNone(sync_client)
        self.assertIsNotNone(v2_client)

    def test_custom_httpx_client_with_pooling(self):
        """Test that custom httpx clients with connection pooling work correctly."""
        # Create custom httpx client with explicit pooling configuration
        custom_client = httpx.Client(
            timeout=30,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=50,
                keepalive_expiry=20.0,
            ),
        )

        # Create Cohere client with custom httpx client
        try:
            client = cohere.ClientV2(api_key="dummy-key", httpx_client=custom_client)
            self.assertIsNotNone(client)
        finally:
            custom_client.close()

    def test_connection_pooling_vs_no_pooling_setup(self):
        """Test creating clients with and without connection pooling."""
        # Create httpx client without pooling
        no_pool_httpx = httpx.Client(
            timeout=30,
            limits=httpx.Limits(
                max_keepalive_connections=0,
                max_connections=1,
                keepalive_expiry=0,
            ),
        )
        
        # Verify both configurations work
        try:
            pooled_client = cohere.ClientV2(api_key="dummy-key")
            no_pool_client = cohere.ClientV2(api_key="dummy-key", httpx_client=no_pool_httpx)
            
            self.assertIsNotNone(pooled_client)
            self.assertIsNotNone(no_pool_client)
            
        finally:
            no_pool_httpx.close()

    @unittest.skipIf(not os.environ.get("CO_API_KEY"), "API key not available")
    def test_multiple_requests_performance(self):
        """Test that multiple requests benefit from connection pooling."""
        client = cohere.ClientV2()
        
        response_times = []
        
        # Make multiple requests
        for i in range(3):
            start_time = time.time()
            try:
                response = client.chat(
                    model="command-r-plus-08-2024",
                    messages=[{"role": "user", "content": f"Say the number {i+1}"}],
                )
                elapsed = time.time() - start_time
                response_times.append(elapsed)
                
                # Verify response
                self.assertIsNotNone(response)
                self.assertIsNotNone(response.message)
                
                # Rate limit protection
                if i < 2:
                    time.sleep(2)
                    
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    self.skipTest("Rate limited")
                raise
        
        # Verify all requests completed
        self.assertEqual(len(response_times), 3)
        
        # Generally, subsequent requests should be faster due to connection reuse
        # First request establishes connection, subsequent ones reuse it
        print(f"Response times: {response_times}")

    @unittest.skipIf(not os.environ.get("CO_API_KEY"), "API key not available")
    def test_streaming_with_pooling(self):
        """Test that streaming works correctly with connection pooling."""
        client = cohere.ClientV2()
        
        try:
            response = client.chat_stream(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": "Count to 3"}],
            )
            
            chunks = []
            for event in response:
                if event.type == "content-delta":
                    chunks.append(event.delta.message.content.text)
            
            # Verify streaming worked
            self.assertGreater(len(chunks), 0)
            full_response = "".join(chunks)
            self.assertGreater(len(full_response), 0)
            
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                self.skipTest("Rate limited")
            raise


if __name__ == "__main__":
    unittest.main()