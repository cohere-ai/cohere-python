import os
import typing
import unittest

import cohere
from cohere import ToolMessage, UserMessage, AssistantMessage

class TestClientInit(unittest.TestCase):
    def test_inits(self) -> None:
        cohere.BedrockClient()
        cohere.BedrockClientV2()
        cohere.SagemakerClient()
        cohere.SagemakerClientV2()
        cohere.Client(api_key="n/a")
        cohere.ClientV2(api_key="n/a")

