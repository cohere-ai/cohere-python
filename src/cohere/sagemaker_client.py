import typing

from .aws_client import AwsClient
from .manually_maintained.cohere_aws.client import Client


class SagemakerClient(AwsClient):
    finetuning: Client

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
            service="sagemaker",
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
            timeout=timeout,
        )
        self.finetuning = Client(region_name=self._aws_region)