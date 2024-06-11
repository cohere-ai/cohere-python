import os
import unittest

import typing
import cohere
from parameterized import parameterized_class  # type: ignore

package_dir = os.path.dirname(os.path.abspath(__file__))
embed_job = os.path.join(package_dir, 'embed_job.jsonl')


model_mapping = {
    "bedrock": {
        "chat_model": "cohere.command-r-plus-v1:0",
        "embed_model": "cohere.embed-multilingual-v3",
        "generate_model": "cohere.command-text-v14",
    },
    "sagemaker": {
        "chat_model": "cohere.command-r-plus-v1:0",
        "embed_model": "cohere.embed-multilingual-v3",
        "generate_model": "cohere-command-light",
        "rerank_model": "rerank",
    },
}


@parameterized_class([
    {
        "platform": "bedrock",
        "client": cohere.BedrockClient(
            timeout=10000,
            aws_region="us-east-1",
            aws_access_key="...",
            aws_secret_key="...",
            aws_session_token="...",
        ),
        "models": model_mapping["bedrock"],
    },
    {
        "platform": "sagemaker",
        "client": cohere.SagemakerClient(
            timeout=10000,
            aws_region="us-east-1",
            aws_access_key="...",
            aws_secret_key="...",
            aws_session_token="...",
        ),
        "models": model_mapping["sagemaker"],
    }
])
@unittest.skip("skip tests until they work in CI")
class TestClient(unittest.TestCase):
    platform: str
    client: cohere.AwsClient
    models: typing.Dict[str, str]

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
