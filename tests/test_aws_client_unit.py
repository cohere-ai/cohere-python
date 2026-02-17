"""
Unit tests (mocked, no AWS credentials needed) for AWS client fixes.

Covers:
- Fix 1: SigV4 signing uses the correct host header after URL rewrite
- Fix 2: cohere_aws.Client conditionally initializes based on mode
- Fix 3: embed() accepts and passes output_dimension and embedding_types
"""

import inspect
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from cohere.manually_maintained.cohere_aws.mode import Mode


class TestSigV4HostHeader(unittest.TestCase):
    """Fix 1: The headers dict passed to AWSRequest for SigV4 signing must
    contain the rewritten Bedrock/SageMaker host, not the stale api.cohere.com."""

    def test_sigv4_signs_with_correct_host(self) -> None:
        captured_aws_request_kwargs: dict = {}

        mock_aws_request_cls = MagicMock()

        def capture_aws_request(**kwargs):  # type: ignore
            captured_aws_request_kwargs.update(kwargs)
            mock_req = MagicMock()
            mock_req.prepare.return_value = MagicMock(
                headers={"host": "bedrock-runtime.us-east-1.amazonaws.com"}
            )
            return mock_req

        mock_aws_request_cls.side_effect = capture_aws_request

        mock_botocore = MagicMock()
        mock_botocore.awsrequest.AWSRequest = mock_aws_request_cls
        mock_botocore.auth.SigV4Auth.return_value = MagicMock()

        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_session.get_credentials.return_value = MagicMock()
        mock_boto3.Session.return_value = mock_session

        with patch("cohere.aws_client.lazy_botocore", return_value=mock_botocore), \
             patch("cohere.aws_client.lazy_boto3", return_value=mock_boto3):

            from cohere.aws_client import map_request_to_bedrock

            hook = map_request_to_bedrock(service="bedrock", aws_region="us-east-1")

            request = httpx.Request(
                method="POST",
                url="https://api.cohere.com/v1/chat",
                headers={"connection": "keep-alive"},
                json={"model": "cohere.command-r-plus-v1:0", "message": "hello"},
            )

            self.assertEqual(request.url.host, "api.cohere.com")

            hook(request)

            self.assertIn("bedrock-runtime.us-east-1.amazonaws.com", str(request.url))

            signed_headers = captured_aws_request_kwargs["headers"]
            self.assertEqual(
                signed_headers["host"],
                "bedrock-runtime.us-east-1.amazonaws.com",
            )


class TestModeConditionalInit(unittest.TestCase):
    """Fix 2: cohere_aws.Client should initialize different boto3 clients
    depending on mode, and default to SAGEMAKER for backwards compat."""

    def test_sagemaker_mode_creates_sagemaker_clients(self) -> None:
        mock_boto3 = MagicMock()
        mock_sagemaker = MagicMock()

        with patch("cohere.manually_maintained.cohere_aws.client.lazy_boto3", return_value=mock_boto3), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_sagemaker", return_value=mock_sagemaker), \
             patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):

            from cohere.manually_maintained.cohere_aws.client import Client

            client = Client(aws_region="us-east-1")

            self.assertEqual(client.mode, Mode.SAGEMAKER)

            service_names = [c[0][0] for c in mock_boto3.client.call_args_list]
            self.assertIn("sagemaker-runtime", service_names)
            self.assertIn("sagemaker", service_names)
            self.assertNotIn("bedrock-runtime", service_names)
            self.assertNotIn("bedrock", service_names)

            mock_sagemaker.Session.assert_called_once()

    def test_bedrock_mode_creates_bedrock_clients(self) -> None:
        mock_boto3 = MagicMock()
        mock_sagemaker = MagicMock()

        with patch("cohere.manually_maintained.cohere_aws.client.lazy_boto3", return_value=mock_boto3), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_sagemaker", return_value=mock_sagemaker), \
             patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-west-2"}):

            from cohere.manually_maintained.cohere_aws.client import Client

            client = Client(aws_region="us-west-2", mode=Mode.BEDROCK)

            self.assertEqual(client.mode, Mode.BEDROCK)

            service_names = [c[0][0] for c in mock_boto3.client.call_args_list]
            self.assertIn("bedrock-runtime", service_names)
            self.assertIn("bedrock", service_names)
            self.assertNotIn("sagemaker-runtime", service_names)
            self.assertNotIn("sagemaker", service_names)

            mock_sagemaker.Session.assert_not_called()

    def test_default_mode_is_sagemaker(self) -> None:
        from cohere.manually_maintained.cohere_aws.client import Client

        sig = inspect.signature(Client.__init__)
        self.assertEqual(sig.parameters["mode"].default, Mode.SAGEMAKER)


class TestEmbedV4Params(unittest.TestCase):
    """Fix 3: embed() should accept output_dimension and embedding_types,
    pass them through to the request body, and strip them when None."""

    @staticmethod
    def _make_bedrock_client():  # type: ignore
        mock_boto3 = MagicMock()
        mock_botocore = MagicMock()
        captured_body: dict = {}

        def fake_invoke_model(**kwargs):  # type: ignore
            captured_body.update(json.loads(kwargs["body"]))
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({"embeddings": [[0.1, 0.2]]}).encode()
            return {"body": mock_body}

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.side_effect = fake_invoke_model

        def fake_boto3_client(service_name, **kwargs):  # type: ignore
            if service_name == "bedrock-runtime":
                return mock_bedrock_client
            return MagicMock()

        mock_boto3.client.side_effect = fake_boto3_client
        return mock_boto3, mock_botocore, captured_body

    def test_embed_accepts_new_params(self) -> None:
        from cohere.manually_maintained.cohere_aws.client import Client

        sig = inspect.signature(Client.embed)
        self.assertIn("output_dimension", sig.parameters)
        self.assertIn("embedding_types", sig.parameters)
        self.assertIsNone(sig.parameters["output_dimension"].default)
        self.assertIsNone(sig.parameters["embedding_types"].default)

    def test_embed_passes_params_to_bedrock(self) -> None:
        mock_boto3, mock_botocore, captured_body = self._make_bedrock_client()

        with patch("cohere.manually_maintained.cohere_aws.client.lazy_boto3", return_value=mock_boto3), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_botocore", return_value=mock_botocore), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_sagemaker", return_value=MagicMock()), \
             patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):

            from cohere.manually_maintained.cohere_aws.client import Client

            client = Client(aws_region="us-east-1", mode=Mode.BEDROCK)
            client.embed(
                texts=["hello world"],
                input_type="search_document",
                model_id="cohere.embed-english-v3",
                output_dimension=256,
                embedding_types=["float", "int8"],
            )

            self.assertEqual(captured_body["output_dimension"], 256)
            self.assertEqual(captured_body["embedding_types"], ["float", "int8"])

    def test_embed_omits_none_params(self) -> None:
        mock_boto3, mock_botocore, captured_body = self._make_bedrock_client()

        with patch("cohere.manually_maintained.cohere_aws.client.lazy_boto3", return_value=mock_boto3), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_botocore", return_value=mock_botocore), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_sagemaker", return_value=MagicMock()), \
             patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):

            from cohere.manually_maintained.cohere_aws.client import Client

            client = Client(aws_region="us-east-1", mode=Mode.BEDROCK)
            client.embed(
                texts=["hello world"],
                input_type="search_document",
                model_id="cohere.embed-english-v3",
            )

            self.assertNotIn("output_dimension", captured_body)
            self.assertNotIn("embedding_types", captured_body)

    def test_embed_with_embedding_types_returns_dict(self) -> None:
        """When embedding_types is specified, the API returns embeddings as a dict.
        The client should return that dict rather than wrapping it in Embeddings."""
        mock_boto3 = MagicMock()
        mock_botocore = MagicMock()

        by_type_embeddings = {"float": [[0.1, 0.2]], "int8": [[1, 2]]}

        def fake_invoke_model(**kwargs):  # type: ignore
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                "embeddings": by_type_embeddings,
                "response_type": "embeddings_by_type",
            }).encode()
            return {"body": mock_body}

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.side_effect = fake_invoke_model

        def fake_boto3_client(service_name, **kwargs):  # type: ignore
            if service_name == "bedrock-runtime":
                return mock_bedrock_client
            return MagicMock()

        mock_boto3.client.side_effect = fake_boto3_client

        with patch("cohere.manually_maintained.cohere_aws.client.lazy_boto3", return_value=mock_boto3), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_botocore", return_value=mock_botocore), \
             patch("cohere.manually_maintained.cohere_aws.client.lazy_sagemaker", return_value=MagicMock()), \
             patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):

            from cohere.manually_maintained.cohere_aws.client import Client

            client = Client(aws_region="us-east-1", mode=Mode.BEDROCK)
            result = client.embed(
                texts=["hello world"],
                input_type="search_document",
                model_id="cohere.embed-english-v3",
                embedding_types=["float", "int8"],
            )

            self.assertIsInstance(result, dict)
            self.assertEqual(result, by_type_embeddings)
