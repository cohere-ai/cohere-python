import os
import unittest

import typing
import cohere
from .test_aws_client import TestClient

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
aws_region = os.getenv("AWS_REGION")
endpoint_type = os.getenv("ENDPOINT_TYPE")


@unittest.skipIf(os.getenv("TEST_AWS"), "tests skipped because TEST_AWS is not set")
class BedrockTestClient(TestClient):
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