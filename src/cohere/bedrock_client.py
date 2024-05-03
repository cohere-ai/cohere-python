import json
import os
import re
import base64
import httpcore
from httpx import URL, SyncByteStream
from tokenizers import Tokenizer  # type: ignore

from . import GenerateStreamText
from .client import Client, ClientEnvironment

import typing

import httpx
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.eventstream import EventStream


class BedrockClient(Client):
    def __init__(
            self,
            *,
            aws_access_key: typing.Optional[str] = None,
            aws_secret_key: typing.Optional[str] = None,
            aws_session_token: typing.Optional[str] = None,
            aws_region: typing.Optional[str] = None,
            base_url: typing.Optional[str] = os.getenv("CO_API_URL"),
            client_name: typing.Optional[str] = None,
            timeout: typing.Optional[float] = None,
    ):
        Client.__init__(
            self,
            base_url=base_url,
            environment=ClientEnvironment.PRODUCTION,
            client_name=client_name,
            timeout=timeout,
            api_key="n/a",
            httpx_client=httpx.Client(
                event_hooks=get_event_hooks(
                    service="bedrock",
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    aws_session_token=aws_session_token,
                    aws_region=aws_region,
                ),
                timeout=timeout,
            ),
        )


EventHook = typing.Callable[..., typing.Any]


def get_event_hooks(
        service: str,
        aws_access_key: typing.Optional[str] = None,
        aws_secret_key: typing.Optional[str] = None,
        aws_session_token: typing.Optional[str] = None,
        aws_region: typing.Optional[str] = None,
) -> typing.Dict[str, typing.List[EventHook]]:
    return {
        "request": [
            map_request_to_bedrock(
                service=service,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                aws_session_token=aws_session_token,
                aws_region=aws_region,
            ),
        ],
        "response": [
            map_response_from_bedrock
        ],
    }


# {'is_finished': True, 'event_type': 'stream-end', 'finish_reason': 'COMPLETE', 'amazon-bedrock-invocationMetrics': {'inputTokenCount': 8, 'outputTokenCount': 490, 'invocationLatency': 15094, 'firstByteLatency': 389}}


TextGeneration = typing.TypedDict('TextGeneration',
                                  {"text": str, "is_finished": str, "event_type": typing.Literal["text-generation"]})
StreamEnd = typing.TypedDict('StreamEnd',
                             {"is_finished": str, "event_type": typing.Literal["stream-end"], "finish_reason": str,
                              "amazon-bedrock-invocationMetrics": typing.TypedDict('InvocationMetrics', {
                                  "inputTokenCount": int, "outputTokenCount": int, "invocationLatency": int,
                                  "firstByteLatency": int})})


class Streamer(SyncByteStream):
    lines: typing.List[bytes]

    def __init__(self, lines: typing.List[bytes]):
        self.lines = lines

    def __iter__(self) -> typing.Iterator[bytes]:
        return iter(self.lines)



def map_response_from_bedrock(
        response: httpx.Response,
) -> httpx.Response:
    output = ""
    regex = r"{[^\}]*}"
    for _text in response.iter_lines():
        match = re.search(regex, _text)
        if match:
            obj = json.loads(match.group())
            if "bytes" in obj:
                # base64 decode the bytes
                str = base64.b64decode(obj["bytes"]).decode("utf-8")
                streamed_obj = json.loads(str)
                if streamed_obj["event_type"] == "text-generation":
                    output += json.dumps(GenerateStreamText.parse_obj(streamed_obj).dict()) + "\n"


    response.stream = Streamer([output.encode("utf-8")])
    response.is_stream_consumed = False
    response.is_closed = False

def map_request_to_bedrock(
        service: str,
        aws_access_key: typing.Optional[str] = None,
        aws_secret_key: typing.Optional[str] = None,
        aws_session_token: typing.Optional[str] = None,
        aws_region: typing.Optional[str] = None,
) -> EventHook:
    session = boto3.Session(
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
    )
    credentials = session.get_credentials()
    signer = SigV4Auth(credentials, service, session.region_name)

    def _event_hook(request: httpx.Request) -> None:
        headers = request.headers.copy()
        del headers["connection"]

        url = get_url(platform=service, aws_region=aws_region)
        request.url = URL(url)
        request.headers["host"] = request.url.host

        aws_request = AWSRequest(
            method=request.method,
            url=url,
            headers=headers,
            data=request.read(),
        )
        signer.add_auth(aws_request)

        request.headers = httpx.Headers(aws_request.prepare().headers)

    return _event_hook


def get_url(
        *,
        platform: str = "bedrock",
        aws_region: typing.Optional[str] = "us-east-1"
) -> str:
    return f"https://{platform}-runtime.{aws_region}.amazonaws.com/model/cohere.command-text-v14/invoke-with-response-stream"
