import os
import unittest

import typing
import cohere

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
aws_region = os.getenv("AWS_REGION")
endpoint_type = os.getenv("ENDPOINT_TYPE")


def _setup_boto3_env():
    """Bridge custom test env vars to standard boto3 credential env vars."""
    if aws_access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
    if aws_secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
    if aws_session_token:
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token


@unittest.skipIf(None == os.getenv("TEST_AWS"), "tests skipped because TEST_AWS is not set")
class TestClient(unittest.TestCase):
    platform: str = "bedrock"
    client: cohere.AwsClient = cohere.BedrockClient(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region,
    )
    models: typing.Dict[str, str] = {
        "chat_model": "cohere.command-r-plus-v1:0",
        "embed_model": "cohere.embed-multilingual-v3",
        "generate_model": "cohere.command-text-v14",
    }

    def test_rerank(self) -> None:
        if self.platform != "sagemaker":
            self.skipTest("Only sagemaker supports rerank")

        docs = [
            'Carson City is the capital city of the American state of Nevada.',
            'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
            'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
            'Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.']

        response = self.client.rerank(
            model=self.models["rerank_model"],
            query='What is the capital of the United States?',
            documents=docs,
            top_n=3,
        )

        self.assertEqual(len(response.results), 3)

    def test_embed(self) -> None:
        response = self.client.embed(
            model=self.models["embed_model"],
            texts=["I love Cohere!"],
            input_type="search_document",
        )
        print(response)

    def test_generate(self) -> None:
        response = self.client.generate(
            model=self.models["generate_model"],
            prompt='Please explain to me how LLMs work',
        )
        print(response)

    def test_generate_stream(self) -> None:
        response = self.client.generate_stream(
            model=self.models["generate_model"],
            prompt='Please explain to me how LLMs work',
        )
        for event in response:
            print(event)
            if event.event_type == "text-generation":
                print(event.text, end='')

    def test_chat(self) -> None:
        response = self.client.chat(
            model=self.models["chat_model"],
            message='Please explain to me how LLMs work',
        )
        print(response)

        self.assertIsNotNone(response.text)
        self.assertIsNotNone(response.generation_id)
        self.assertIsNotNone(response.finish_reason)
        
        self.assertIsNotNone(response.meta)
        if response.meta is not None:
            self.assertIsNotNone(response.meta.tokens)
            if response.meta.tokens is not None:
                self.assertIsNotNone(response.meta.tokens.input_tokens)
                self.assertIsNotNone(response.meta.tokens.output_tokens)

            self.assertIsNotNone(response.meta.billed_units)
            if response.meta.billed_units is not None:
                self.assertIsNotNone(response.meta.billed_units.input_tokens)
                self.assertIsNotNone(response.meta.billed_units.input_tokens)

    def test_chat_stream(self) -> None:
        response_types = set()
        response = self.client.chat_stream(
            model=self.models["chat_model"],
            message='Please explain to me how LLMs work',
        )
        for event in response:
            response_types.add(event.event_type)
            if event.event_type == "text-generation":
                print(event.text, end='')
                self.assertIsNotNone(event.text)
            if event.event_type == "stream-end":
                self.assertIsNotNone(event.finish_reason)
                self.assertIsNotNone(event.response)
                self.assertIsNotNone(event.response.text)

        self.assertSetEqual(response_types, {"text-generation", "stream-end"})


@unittest.skipIf(None == os.getenv("TEST_AWS"), "tests skipped because TEST_AWS is not set")
class TestBedrockClientV2(unittest.TestCase):
    """Integration tests for BedrockClientV2 (httpx-based).

    Fix 1 validation: If these pass, SigV4 signing uses the correct host header,
    since the request would fail with a signature mismatch otherwise.
    """

    client: cohere.ClientV2 = cohere.BedrockClientV2(
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region,
    )

    def test_embed(self) -> None:
        response = self.client.embed(
            model="cohere.embed-multilingual-v3",
            texts=["I love Cohere!"],
            input_type="search_document",
            embedding_types=["float"],
        )
        self.assertIsNotNone(response)

    def test_embed_with_output_dimension(self) -> None:
        response = self.client.embed(
            model="cohere.embed-english-v3",
            texts=["I love Cohere!"],
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=256,
        )
        self.assertIsNotNone(response)


@unittest.skipIf(None == os.getenv("TEST_AWS"), "tests skipped because TEST_AWS is not set")
class TestCohereAwsBedrockClient(unittest.TestCase):
    """Integration tests for cohere_aws.Client in Bedrock mode (boto3-based).

    Validates:
    - Fix 2: Client can be initialized with mode=BEDROCK without importing sagemaker
    - Fix 3: embed() accepts output_dimension and embedding_types
    """
    client: typing.Any = None

    @classmethod
    def setUpClass(cls) -> None:
        _setup_boto3_env()
        from cohere.manually_maintained.cohere_aws.client import Client
        from cohere.manually_maintained.cohere_aws.mode import Mode
        cls.client = Client(aws_region=aws_region, mode=Mode.BEDROCK)

    def test_client_is_bedrock_mode(self) -> None:
        from cohere.manually_maintained.cohere_aws.mode import Mode
        self.assertEqual(self.client.mode, Mode.BEDROCK)

    def test_embed(self) -> None:
        response = self.client.embed(
            texts=["I love Cohere!"],
            input_type="search_document",
            model_id="cohere.embed-multilingual-v3",
        )
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
        self.assertGreater(len(response.embeddings), 0)

    def test_embed_with_embedding_types(self) -> None:
        response = self.client.embed(
            texts=["I love Cohere!"],
            input_type="search_document",
            model_id="cohere.embed-multilingual-v3",
            embedding_types=["float"],
        )
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_embed_with_output_dimension(self) -> None:
        response = self.client.embed(
            texts=["I love Cohere!"],
            input_type="search_document",
            model_id="cohere.embed-english-v3",
            output_dimension=256,
            embedding_types=["float"],
        )
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)

    def test_embed_without_new_params(self) -> None:
        """Backwards compat: embed() still works without the new v4 params."""
        response = self.client.embed(
            texts=["I love Cohere!"],
            input_type="search_document",
            model_id="cohere.embed-multilingual-v3",
        )
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.embeddings)
