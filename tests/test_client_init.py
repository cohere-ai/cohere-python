import os
import typing
import unittest

import cohere
from cohere import ToolMessage, UserMessage, AssistantMessage

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

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

