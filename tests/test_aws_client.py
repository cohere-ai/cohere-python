import os
import unittest

import cohere
from parameterized import parameterized_class  # type: ignore

package_dir = os.path.dirname(os.path.abspath(__file__))
embed_job = os.path.join(package_dir, 'embed_job.jsonl')


@parameterized_class([
    {
        "client": cohere.BedrockClient(
            timeout=10000,
            aws_region="us-east-1",
            chat_model="cohere.command-r-plus-v1:0",
            embed_model="cohere.embed-multilingual-v3",
            generate_model="cohere.command-text-v14",
            aws_access_key="...",
            aws_secret_key="...",
            aws_session_token="...",
        )
    },
    {
        "client": cohere.SagemakerClient(
            timeout=10000,
            aws_region="us-east-1",
            chat_model="cohere.command-r-plus-v1:0",
            embed_model="cohere.embed-multilingual-v3",
            generate_model="cohere-command-light",
            aws_access_key="...",
            aws_secret_key="...",
            aws_session_token="...",
        )
    }
])
@unittest.skip("skip tests until they work in CI")
class TestClient(unittest.TestCase):
    client: cohere.AwsClient;

    def test_embed(self) -> None:
        response = self.client.embed(
            texts=["I love Cohere!"],
            input_type="search_document",
        )
        print(response)

    def test_generate(self) -> None:
        response = self.client.generate(
            prompt='Please explain to me how LLMs work',
        )
        print(response)

    def test_generate_stream(self) -> None:
        response = self.client.generate_stream(
            prompt='Please explain to me how LLMs work',
        )
        for event in response:
            print(event)
            if event.event_type == "text-generation":
                print(event.text, end='')

    def test_chat(self) -> None:
        response = self.client.chat(
            message='Please explain to me how LLMs work',
        )
        print(response)

        self.assertIsNotNone(response.text)
        self.assertIsNotNone(response.generation_id)
        self.assertIsNotNone(response.finish_reason)

    def test_chat_stream(self) -> None:
        response_types = set()
        response = self.client.chat_stream(
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
