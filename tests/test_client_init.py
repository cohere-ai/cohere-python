import os
import typing
import unittest

import cohere
from cohere import ToolMessage, UserMessage, AssistantMessage

import importlib.util
HAS_BOTO3 = importlib.util.find_spec("boto3") is not None

class TestClientInit(unittest.TestCase):
    @unittest.skipUnless(HAS_BOTO3, "boto3 not installed")
    def test_aws_inits(self) -> None:
        cohere.BedrockClient()
        cohere.BedrockClientV2()
        cohere.SagemakerClient()
        cohere.SagemakerClientV2()

    def test_inits(self) -> None:
        cohere.Client(api_key="n/a")
        cohere.ClientV2(api_key="n/a")

