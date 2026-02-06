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

    @unittest.skip(
        "OCI on-demand models don't support multiple embedding types in a single call. "
        "The embedding_types parameter in OCI accepts a single value, not a list."
    )
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

    @unittest.skip(
        "OCI TEXT_GENERATION models are finetune base models, not available via on-demand inference. "
        "Only CHAT models (command-r, command-a) support on-demand inference on OCI."
    )
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

    @unittest.skip(
        "OCI TEXT_GENERATION models are finetune base models, not available via on-demand inference. "
        "Only CHAT models (command-r, command-a) support on-demand inference on OCI."
    )
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

    @unittest.skip(
        "OCI TEXT_RERANK models are base models, not available via on-demand inference. "
        "These models require fine-tuning and deployment before use on OCI."
    )
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

    def test_embed_v2(self):
        """Test embedding with v2 client."""
        response = self.client.embed(
            model="embed-english-v3.0",
            texts=["Hello from v2", "Second text"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        # V2 returns embeddings as a dict with "float" key
        self.assertIsNotNone(response.embeddings.float_)
        self.assertEqual(len(response.embeddings.float_), 2)
        # Verify embedding dimensions (1024 for embed-english-v3.0)
        self.assertEqual(len(response.embeddings.float_[0]), 1024)

    def test_embed_with_model_prefix_v2(self):
        """Test embedding with 'cohere.' model prefix on v2 client."""
        response = self.client.embed(
            model="cohere.embed-english-v3.0",
            texts=["Test with prefix"],
            input_type="search_document",
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertIsNotNone(response.embeddings.float_)
        self.assertEqual(len(response.embeddings.float_), 1)

    def test_chat_v2(self):
        """Test chat with v2 client."""
        response = self.client.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)

    @unittest.skip(
        "Command A Reasoning model (command-a-reasoning-08-2025) may not be available in all regions. "
        "Enable this test when the reasoning model is available in your OCI region."
    )
    def test_chat_v2_with_thinking(self):
        """Test chat with thinking parameter for Command A Reasoning model."""
        from cohere.types import Thinking

        response = self.client.chat(
            model="command-a-reasoning-08-2025",
            messages=[{"role": "user", "content": "What is 15 * 27? Think step by step."}],
            thinking=Thinking(type="enabled", token_budget=5000),
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)
        # The response should contain content (may include thinking content)
        self.assertIsNotNone(response.message.content)

    @unittest.skip(
        "Command A Reasoning model (command-a-reasoning-08-2025) may not be available in all regions. "
        "Enable this test when the reasoning model is available in your OCI region."
    )
    def test_chat_stream_v2_with_thinking(self):
        """Test streaming chat with thinking parameter for Command A Reasoning model."""
        from cohere.types import Thinking

        events = []
        for event in self.client.chat_stream(
            model="command-a-reasoning-08-2025",
            messages=[{"role": "user", "content": "What is 15 * 27? Think step by step."}],
            thinking=Thinking(type="enabled", token_budget=5000),
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)
        # Verify we received content-delta events
        content_delta_events = [e for e in events if hasattr(e, "type") and e.type == "content-delta"]
        self.assertTrue(len(content_delta_events) > 0)

    def test_chat_stream_v2(self):
        """Test streaming chat with v2 client."""
        events = []
        for event in self.client.chat_stream(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Count from 1 to 3"}],
        ):
            events.append(event)

        self.assertTrue(len(events) > 0)
        # Verify we received content-delta events with text
        content_delta_events = [e for e in events if hasattr(e, "type") and e.type == "content-delta"]
        self.assertTrue(len(content_delta_events) > 0)

        # Verify we can extract text from events
        full_text = ""
        for event in events:
            if (
                hasattr(event, "delta")
                and event.delta
                and hasattr(event.delta, "message")
                and event.delta.message
                and hasattr(event.delta.message, "content")
                and event.delta.message.content
                and hasattr(event.delta.message.content, "text")
            ):
                full_text += event.delta.message.content.text

        # Should have received some text
        self.assertTrue(len(full_text) > 0)

    @unittest.skip(
        "OCI TEXT_RERANK models are base models, not available via on-demand inference. "
        "These models require fine-tuning and deployment before use on OCI."
    )
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

    @unittest.skip(
        "OCI TEXT_RERANK models are base models, not available via on-demand inference. "
        "These models require fine-tuning and deployment before use on OCI."
    )
    def test_rerank_v3(self):
        """Test rerank-english-v3.0 model."""
        response = self.client.rerank(
            model="rerank-english-v3.1",
            query="AI",
            documents=["Artificial Intelligence", "Biology"],
        )
        self.assertIsNotNone(response.results)


class TestOciClientTransformations(unittest.TestCase):
    """Unit tests for OCI request/response transformations (no OCI credentials required)."""

    def test_thinking_parameter_transformation(self):
        """Test that thinking parameter is correctly transformed to OCI format."""
        from cohere.oci_client import transform_request_to_oci

        cohere_body = {
            "model": "command-a-reasoning-08-2025",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "thinking": {
                "type": "enabled",
                "token_budget": 10000,
            },
        }

        result = transform_request_to_oci("chat", cohere_body, "compartment-123")

        # Verify thinking parameter is transformed
        chat_request = result["chatRequest"]
        self.assertIn("thinking", chat_request)
        self.assertEqual(chat_request["thinking"]["type"], "ENABLED")
        self.assertEqual(chat_request["thinking"]["token_budget"], 10000)

    def test_thinking_parameter_disabled(self):
        """Test that disabled thinking is correctly transformed."""
        from cohere.oci_client import transform_request_to_oci

        cohere_body = {
            "model": "command-a-reasoning-08-2025",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {
                "type": "disabled",
            },
        }

        result = transform_request_to_oci("chat", cohere_body, "compartment-123")

        chat_request = result["chatRequest"]
        self.assertIn("thinking", chat_request)
        self.assertEqual(chat_request["thinking"]["type"], "DISABLED")
        self.assertNotIn("token_budget", chat_request["thinking"])

    def test_thinking_response_transformation(self):
        """Test that thinking content in response is correctly transformed."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [
                        {"type": "THINKING", "thinking": "Let me think about this..."},
                        {"type": "TEXT", "text": "The answer is 4."},
                    ],
                },
                "finishReason": "COMPLETE",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # Verify content types are lowercased
        self.assertEqual(result["message"]["content"][0]["type"], "thinking")
        self.assertEqual(result["message"]["content"][1]["type"], "text")

    def test_stream_event_thinking_transformation(self):
        """Test that thinking content in stream events is correctly transformed."""
        from cohere.oci_client import transform_stream_event

        # OCI thinking event
        oci_event = {
            "message": {
                "content": [{"type": "THINKING", "thinking": "Reasoning step..."}]
            }
        }

        result = transform_stream_event("chat", oci_event, is_v2=True)

        self.assertEqual(result["type"], "content-delta")
        self.assertIn("thinking", result["delta"]["message"]["content"])
        self.assertEqual(result["delta"]["message"]["content"]["thinking"], "Reasoning step...")

    def test_stream_event_text_transformation(self):
        """Test that text content in stream events is correctly transformed."""
        from cohere.oci_client import transform_stream_event

        # OCI text event
        oci_event = {
            "message": {
                "content": [{"type": "TEXT", "text": "The answer is..."}]
            }
        }

        result = transform_stream_event("chat", oci_event, is_v2=True)

        self.assertEqual(result["type"], "content-delta")
        self.assertIn("text", result["delta"]["message"]["content"])
        self.assertEqual(result["delta"]["message"]["content"]["text"], "The answer is...")

    def test_thinking_parameter_none(self):
        """Test that thinking=None does not crash (issue: null guard)."""
        from cohere.oci_client import transform_request_to_oci

        cohere_body = {
            "model": "command-a-03-2025",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": None,  # Explicitly set to None
        }

        # Should not crash with TypeError
        result = transform_request_to_oci("chat", cohere_body, "compartment-123")

        chat_request = result["chatRequest"]
        # thinking should not be in request when None
        self.assertNotIn("thinking", chat_request)

    def test_v2_response_role_lowercased(self):
        """Test that V2 response message role is lowercased."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "Hello"}],
                },
                "finishReason": "COMPLETE",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # Role should be lowercased
        self.assertEqual(result["message"]["role"], "assistant")

    def test_v2_response_finish_reason_uppercase(self):
        """Test that V2 response finish_reason stays uppercase."""
        from cohere.oci_client import transform_oci_response_to_cohere

        oci_response = {
            "chatResponse": {
                "id": "test-id",
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "Hello"}],
                },
                "finishReason": "MAX_TOKENS",
                "usage": {"inputTokens": 10, "completionTokens": 20},
            }
        }

        result = transform_oci_response_to_cohere("chat", oci_response, is_v2=True)

        # V2 finish_reason should stay uppercase
        self.assertEqual(result["finish_reason"], "MAX_TOKENS")


if __name__ == "__main__":
    unittest.main()
