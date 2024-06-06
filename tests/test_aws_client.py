import os
import unittest

import typing
import cohere
from parameterized import parameterized_class  # type: ignore

package_dir = os.path.dirname(os.path.abspath(__file__))
embed_job = os.path.join(package_dir, 'embed_job.jsonl')


models = {
    "bedrock": {
        "chat_model": "cohere.command-r-plus-v1:0",
        "embed_model": "cohere.embed-multilingual-v3",
        "generate_model": "cohere.command-text-v14",
    },
    "sagemaker": {
        "chat_model": "cohere.command-r-plus-v1:0",
        "embed_model": "cohere.embed-multilingual-v3",
        "generate_model": "cohere-command-light",
    },
}


@parameterized_class([
    {
        "client": cohere.BedrockClient(
            timeout=10000,
            aws_region="us-east-1",
            aws_access_key="...",
            aws_secret_key="...",
            aws_session_token="...",
        ),
        "models": models["bedrock"],
    },
    {
        "client": cohere.SagemakerClient(
            timeout=10000,
            aws_region="us-east-1",
            aws_access_key="...",
            aws_secret_key="...",
            aws_session_token="...",
        ),
        "models": models["sagemaker"],
    }
])
@unittest.skip("skip tests until they work in CI")
class TestClient(unittest.TestCase):
    client: cohere.AwsClient
    models: typing.Dict[str, str]

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
