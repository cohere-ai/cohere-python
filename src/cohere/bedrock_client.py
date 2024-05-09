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

import typing

import httpx
import boto3  # type: ignore
from botocore.auth import SigV4Auth  # type: ignore
from botocore.awsrequest import AWSRequest  # type: ignore

from .core import construct_type, UncheckedBaseModel


class BedrockClient(Client):
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
        Client.__init__(
            self,
            base_url="https://api.cohere.com",  # this url is unused for BedrockClient
            environment=ClientEnvironment.PRODUCTION,
            client_name="n/a",
            timeout=timeout,
            api_key="n/a",
            httpx_client=httpx.Client(
                event_hooks=get_event_hooks(
                    service="bedrock",
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    aws_session_token=aws_session_token,
                    aws_region=aws_region,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    generate_model=generate_model,
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
        chat_model: typing.Optional[str] = None,
        embed_model: typing.Optional[str] = None,
        generate_model: typing.Optional[str] = None,
) -> typing.Dict[str, typing.List[EventHook]]:
    return {
        "request": [
            map_request_to_bedrock(
                service=service,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                aws_session_token=aws_session_token,
                aws_region=aws_region,
                chat_model=chat_model,
                embed_model=embed_model,
                generate_model=generate_model,
            ),
        ],
        "response": [
            map_response_from_bedrock(
                chat_model=chat_model,
                embed_model=embed_model,
                generate_model=generate_model,
            )
        ],
    }


TextGeneration = typing.TypedDict('TextGeneration',
                                  {"text": str, "is_finished": str, "event_type": typing.Literal["text-generation"]})
StreamEnd = typing.TypedDict('StreamEnd',
                             {"is_finished": str, "event_type": typing.Literal["stream-end"], "finish_reason": str,
                              # "amazon-bedrock-invocationMetrics": {
                              #     "inputTokenCount": int, "outputTokenCount": int, "invocationLatency": int,
                              #     "firstByteLatency": int}
                              })


class Streamer(SyncByteStream):
    lines: typing.Iterator[bytes]

    def __init__(self, lines: typing.Iterator[bytes]):
        self.lines = lines

    def __iter__(self) -> typing.Iterator[bytes]:
        return self.lines


response_mapping: typing.Dict[str, typing.Any] = {
    "chat": NonStreamedChatResponse,
    "embed": EmbedResponse,
    "generate": Generation,
}

stream_response_mapping: typing.Dict[str, typing.Any] = {
    "chat": StreamedChatResponse,
    "generate": GenerateStreamedResponse,
}


def stream_generator(response: httpx.Response, endpoint: str) -> typing.Iterator[bytes]:
    regex = r"{[^\}]*}"

    for _text in response.iter_lines():
        match = re.search(regex, _text)
        if match:
            obj = json.loads(match.group())
            if "bytes" in obj:
                base64_payload = base64.b64decode(obj["bytes"]).decode("utf-8")
                streamed_obj = json.loads(base64_payload)
                if "event_type" in streamed_obj:
                    response_type = stream_response_mapping[endpoint]
                    parsed = typing.cast(response_type,  # type: ignore
                                         construct_type(type_=response_type, object_=streamed_obj))  # type: ignore
                    yield (json.dumps(parsed.dict()) + "\n").encode("utf-8")  # type: ignore


def map_response_from_bedrock(
        chat_model: typing.Optional[str] = None,
        embed_model: typing.Optional[str] = None,
        generate_model: typing.Optional[str] = None,
):
    def _hook(
            response: httpx.Response,
    ) -> None:
        stream = response.headers["content-type"] == "application/vnd.amazon.eventstream"
        endpoint = get_endpoint_from_url(response.url.path, chat_model, embed_model, generate_model)
        output: typing.Iterator[bytes]

        if stream:
            output = stream_generator(httpx.Response(
                stream=response.stream,
                status_code=response.status_code,
            ), endpoint)
        else:
            response_type = response_mapping[endpoint]
            output = iter([json.dumps(typing.cast(response_type,  # type: ignore
                                                  construct_type(
                                                      type_=response_type,
                                                      object_=json.loads(response.read()))).dict()  # type: ignore
                                      ).encode(
                "utf-8")])

        response.stream = Streamer(output)
        response.is_stream_consumed = False
        response.is_closed = False

    return _hook


def map_request_to_bedrock(
        service: str,
        aws_access_key: typing.Optional[str] = None,
        aws_secret_key: typing.Optional[str] = None,
        aws_session_token: typing.Optional[str] = None,
        aws_region: typing.Optional[str] = None,
        chat_model: typing.Optional[str] = None,
        embed_model: typing.Optional[str] = None,
        generate_model: typing.Optional[str] = None,
) -> EventHook:
    session = boto3.Session(
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
    )
    credentials = session.get_credentials()
    signer = SigV4Auth(credentials, service, session.region_name)

    model_lookup = {
        "embed": embed_model,
        "chat": chat_model,
        "generate": generate_model,
    }

    def _event_hook(request: httpx.Request) -> None:
        headers = request.headers.copy()
        del headers["connection"]

        endpoint = request.url.path.split("/")[-1]
        body = json.loads(request.read())

        url = get_url(
            platform=service,
            aws_region=aws_region,
            model=model_lookup[endpoint],  # type: ignore
            stream="stream" in body and body["stream"],
        )
        request.url = URL(url)
        request.headers["host"] = request.url.host

        if "stream" in body:
            del body["stream"]

        new_body = json.dumps(body).encode("utf-8")
        request.stream = ByteStream(new_body)
        request._content = new_body
        headers["content-length"] = str(len(new_body))

        aws_request = AWSRequest(
            method=request.method,
            url=url,
            headers=headers,
            data=request.read(),
        )
        signer.add_auth(aws_request)

        request.headers = httpx.Headers(aws_request.prepare().headers)

    return _event_hook


def get_endpoint_from_url(url: str,
                          chat_model: typing.Optional[str] = None,
                          embed_model: typing.Optional[str] = None,
                          generate_model: typing.Optional[str] = None,
                          ) -> str:
    if chat_model and chat_model in url:
        return "chat"
    if embed_model and embed_model in url:
        return "embed"
    if generate_model and generate_model in url:
        return "generate"
    raise ValueError(f"Unknown endpoint in url: {url}")


def get_url(
        *,
        platform: str,
        aws_region: typing.Optional[str],
        model: str,
        stream: bool,
) -> str:
    endpoint = "invoke" if not stream else "invoke-with-response-stream"
    return f"https://{platform}-runtime.{aws_region}.amazonaws.com/model/{model}/{endpoint}"
