"""Integration tests for OCI Generative AI client.

These tests require:
1. OCI SDK installed: pip install oci
2. OCI credentials configured in ~/.oci/config
3. TEST_OCI environment variable set to run
4. OCI_COMPARTMENT_ID environment variable with valid OCI compartment OCID
5. OCI_REGION environment variable (optional, defaults to us-chicago-1)

Run with:
    TEST_OCI=1 OCI_COMPARTMENT_ID=ocid1.compartment.oc1... pytest tests/test_oci_client.py
"""

import os
import unittest

import cohere


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClient(unittest.TestCase):
    """Test OciClient (v1 API) with OCI Generative AI."""

    def setUp(self):
        """Set up OCI client for each test."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        region = os.getenv("OCI_REGION", "us-chicago-1")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        self.client = cohere.OciClient(
            oci_region=region,
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

    def test_embed(self):
        """Test embedding generation with OCI."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Hello world", "Cohere on OCI"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings), 2)
        # Verify embedding dimensions (1024 for embed-english-v3.0)
        self.assertEqual(len(response.embeddings[0]), 1024)

    def test_embed_with_model_prefix(self):
        """Test embedding with 'cohere.' model prefix."""
        response = self.client.embed(
            model="cohere.embed-english-v3.0",
            texts=["Test with prefix"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings), 1)

    @unittest.skip("Multiple embedding types not yet implemented for OCI")
    def test_embed_multiple_types(self):
        """Test embedding with multiple embedding types."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Multi-type test"],
            input_type="search_document",
            embedding_types=["float", "int8"],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_chat(self):
        """Test chat with OCI."""
        response = self.client.chat(
            model="command-r-08-2024",
            message="What is 2+2? Answer with just the number.",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.text)
        self.assertIn("4", response.text)

    def test_chat_with_history(self):
        """Test chat with conversation history."""
        response = self.client.chat(
            model="command-r-08-2024",
            message="What was my previous question?",
            chat_history=[
                {"role": "USER", "message": "What is the capital of France?"},
                {"role": "CHATBOT", "message": "The capital of France is Paris."},
            ],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.text)

    def test_chat_stream(self):
        """Test streaming chat with OCI."""
        events = []
        for event in self.client.chat_stream(
            model="command-r-08-2024",
            message="Count from 1 to 3.",
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)
        # Verify we received text generation events
        text_events = [e for e in events if hasattr(e, "text") and e.text]
        self.assertTrue(len(text_events) > 0)

    @unittest.skip("OCI TEXT_GENERATION models are finetune base models - not callable via on-demand inference")
    def test_generate(self):
        """Test text generation with OCI."""
        response = self.client.generate(
            model="command-r-08-2024",
            prompt="Write a haiku about clouds.",
            max_tokens=100,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.generations)
        self.assertTrue(len(response.generations) > 0)
        self.assertIsNotNone(response.generations[0].text)

    @unittest.skip("OCI TEXT_GENERATION models are finetune base models - not callable via on-demand inference")
    def test_generate_stream(self):
        """Test streaming text generation with OCI."""
        events = []
        for event in self.client.generate_stream(
            model="command-r-08-2024",
            prompt="Say hello",
            max_tokens=20,
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)

    @unittest.skip("OCI TEXT_RERANK models are base models - not callable via on-demand inference")
    def test_rerank(self):
        """Test reranking with OCI."""
        query = "What is the capital of France?"
        documents = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany.",
        ]

        response = self.client.rerank(
            model="rerank-english-v3.1",
            query=query,
            documents=documents,
            top_n=2,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.results)
        self.assertEqual(len(response.results), 2)
        # First result should be the Paris document
        self.assertEqual(response.results[0].index, 0)
        self.assertGreater(response.results[0].relevance_score, 0.5)


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientV2(unittest.TestCase):
    """Test OciClientV2 (v2 API) with OCI Generative AI."""

    def setUp(self):
        """Set up OCI v2 client for each test."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        region = os.getenv("OCI_REGION", "us-chicago-1")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        self.client = cohere.OciClientV2(
            oci_region=region,
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

    @unittest.skip("Embed API is identical in V1 and V2 - use V1 client for embed")
    def test_embed_v2(self):
        """Test embedding with v2 client (same as V1 for embed)."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Hello from v2"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_chat_v2(self):
        """Test chat with v2 client."""
        response = self.client.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)

    @unittest.skip("OCI TEXT_RERANK models are base models - not callable via on-demand inference")
    def test_rerank_v2(self):
        """Test reranking with v2 client."""
        response = self.client.rerank(
            model="rerank-english-v3.1",
            query="What is AI?",
            documents=["AI is artificial intelligence.", "AI is not natural."],
            top_n=1,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.results)


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientAuthentication(unittest.TestCase):
    """Test different OCI authentication methods."""

    def test_config_file_auth(self):
        """Test authentication using OCI config file."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        # Use API_KEY_AUTH profile (DEFAULT may be session-based)
        profile = os.getenv("OCI_PROFILE", "API_KEY_AUTH")
        client = cohere.OciClient(
            oci_region="us-chicago-1",
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

        # Test with a simple embed call
        response = client.embed(
            model="embed-english-v3.0",
            texts=["Auth test"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_custom_profile_auth(self):
        """Test authentication using custom OCI profile."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        client = cohere.OciClient(
            oci_profile=profile,
            oci_region="us-chicago-1",
            oci_compartment_id=compartment_id,
        )

        response = client.embed(
            model="embed-english-v3.0",
            texts=["Profile auth test"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientErrors(unittest.TestCase):
    """Test error handling in OCI client."""

    def test_missing_compartment_id(self):
        """Test error when compartment ID is missing."""
        with self.assertRaises(TypeError):
            cohere.OciClient(
                oci_region="us-chicago-1",
                # Missing oci_compartment_id
            )

    @unittest.skip("Region is available in config file for current test environment")
    def test_missing_region(self):
        """Test error when region is missing and not in config."""
        # This test assumes no region in config file
        # If config has region, this will pass, so we just check it doesn't crash
        try:
            client = cohere.OciClient(
                oci_compartment_id="ocid1.compartment.oc1...",
            )
            # If this succeeds, region was in config
            self.assertIsNotNone(client)
        except ValueError as e:
            # Expected if no region in config
            self.assertIn("region", str(e).lower())

    def test_invalid_model(self):
        """Test error handling with invalid model."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        profile = os.getenv("OCI_PROFILE", "API_KEY_AUTH")
        client = cohere.OciClient(
            oci_region="us-chicago-1",
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

        # OCI should return an error for invalid model
        with self.assertRaises(Exception):
            client.embed(
                model="invalid-model-name",
                texts=["Test"],
                input_type="search_document",
            )


@unittest.skipIf(os.getenv("TEST_OCI") is None, "TEST_OCI not set")
class TestOciClientModels(unittest.TestCase):
    """Test different Cohere models on OCI."""

    def setUp(self):
        """Set up OCI client for each test."""
        compartment_id = os.getenv("OCI_COMPARTMENT_ID")
        if not compartment_id:
            self.skipTest("OCI_COMPARTMENT_ID not set")

        region = os.getenv("OCI_REGION", "us-chicago-1")
        profile = os.getenv("OCI_PROFILE", "DEFAULT")

        self.client = cohere.OciClient(
            oci_region=region,
            oci_compartment_id=compartment_id,
            oci_profile=profile,
        )

    def test_embed_english_v3(self):
        """Test embed-english-v3.0 model."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Test"],
            input_type="search_document",
        )
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0]), 1024)

    def test_embed_light_v3(self):
        """Test embed-english-light-v3.0 model."""
        response = self.client.embed(
            model="embed-english-light-v3.0",
            texts=["Test"],
            input_type="search_document",
        )
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0]), 384)

    def test_embed_multilingual_v3(self):
        """Test embed-multilingual-v3.0 model."""
        response = self.client.embed(
            model="embed-multilingual-v3.0",
            texts=["Test"],
            input_type="search_document",
        )
        self.assertIsNotNone(response.embeddings)
        self.assertEqual(len(response.embeddings[0]), 1024)

    def test_command_r_plus(self):
        """Test command-r-plus model for chat."""
        response = self.client.chat(
            model="command-r-08-2024",
            message="Hello",
        )
        self.assertIsNotNone(response.text)

    @unittest.skip("OCI TEXT_RERANK models are base models - not callable via on-demand inference")
    def test_rerank_v3(self):
        """Test rerank-english-v3.0 model."""
        response = self.client.rerank(
            model="rerank-english-v3.1",
            query="AI",
            documents=["Artificial Intelligence", "Biology"],
        )
        self.assertIsNotNone(response.results)


if __name__ == "__main__":
    unittest.main()
