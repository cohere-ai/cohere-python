import json
import os
import re
import base64
import httpcore
from httpx import URL, SyncByteStream, ByteStream
from tokenizers import Tokenizer  # type: ignore

from . import GenerateStreamText, GenerateStreamEndResponse, GenerateStreamEnd, GenerateStreamedResponse, Generation, \
    NonStreamedChatResponse, EmbedResponse, StreamedChatResponse
from .client import Client, ClientEnvironment
from .aws_client import AwsClient

import typing

class SagemakerClient(AwsClient):
    def __init__(
            self,
            *,
            aws_access_key: typing.Optional[str] = None,
            aws_secret_key: typing.Optional[str] = None,
            aws_session_token: typing.Optional[str] = None,
            aws_region: typing.Optional[str] = None,
            timeout: typing.Optional[float] = None,
            chat_model: typing.Optional[str] = None,
            embed_model: typing.Optional[str] = None,
            generate_model: typing.Optional[str] = None,
    ):
        AwsClient.__init__(
            self,
            service="sagemaker",
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            timeout=timeout,
            chat_model=chat_model,
            embed_model=embed_model,
            generate_model=generate_model,
        )
