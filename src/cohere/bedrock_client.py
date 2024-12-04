import typing

from tokenizers import Tokenizer  # type: ignore

from .aws_client import AwsClient, AwsClientV2


class BedrockClient(AwsClient):
    def __init__(
            self,
            *,
            aws_access_key: typing.Optional[str] = None,
            aws_secret_key: typing.Optional[str] = None,
            aws_session_token: typing.Optional[str] = None,
            aws_region: typing.Optional[str] = None,
            timeout: typing.Optional[float] = None,
    ):
        AwsClient.__init__(
            self,
            service="bedrock",
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            timeout=timeout,
        )

    def rerank(self, *, query, documents, model = ..., top_n = ..., rank_fields = ..., return_documents = ..., max_chunks_per_doc = ..., request_options = None):
        raise NotImplementedError("Please use cohere.BedrockClientV2 instead: Rerank API on Bedrock is not supported with cohere.BedrockClient for this model.")

class BedrockClientV2(AwsClientV2):
    def __init__(
            self,
            *,
            aws_access_key: typing.Optional[str] = None,
            aws_secret_key: typing.Optional[str] = None,
            aws_session_token: typing.Optional[str] = None,
            aws_region: typing.Optional[str] = None,
            timeout: typing.Optional[float] = None,
    ):
        AwsClientV2.__init__(
            self,
            service="bedrock",
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            timeout=timeout,
        )
