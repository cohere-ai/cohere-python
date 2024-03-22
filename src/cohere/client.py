import typing

import httpx

from .base_client import BaseCohere, AsyncBaseCohere
from .environment import ClientEnvironment
from .utils import wait, async_wait
import os

from .overrides import run_overrides

run_overrides()


# Use NoReturn as Never type for compatibility
Never = typing.NoReturn


def validate_args(obj: typing.Any, method_name: str, check_fn: typing.Callable[[typing.Any], typing.Any]) -> None:
    method = getattr(obj, method_name)

    def wrapped(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        check_fn(*args, **kwargs)
        return method(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def throw_if_stream_is_true(*args, **kwargs) -> None:
    if kwargs.get("stream") is True:
        raise ValueError(
            "Since python sdk cohere==5.0.0, you must now use chat_stream(...) instead of chat(stream=True, ...)"
        )


def moved_function(fn_name: str, new_fn_name: str) -> typing.Any:
    """
    This method is moved. Please update usage.
    """

    def fn(*args, **kwargs):
        raise ValueError(
            f"Since python sdk cohere==5.0.0, the function {fn_name}(...) has been moved to {new_fn_name}(...). "
            f"Please update your code. Issues may be filed in https://github.com/cohere-ai/cohere-python/issues."
        )

    return fn


def deprecated_function(fn_name: str) -> typing.Any:
    """
    This method is deprecated. Please update usage.
    """

    def fn(*args, **kwargs):
        raise ValueError(
            f"Since python sdk cohere==5.0.0, the function {fn_name}(...) has been deprecated. "
            f"Please update your code. Issues may be filed in https://github.com/cohere-ai/cohere-python/issues."
        )

    return fn


class Client(BaseCohere):
    def __init__(
            self,
            api_key: typing.Optional[typing.Union[str, typing.Callable[[], str]]] = None,
            *,
            base_url: typing.Optional[str] = os.getenv("CO_API_URL"),
            environment: ClientEnvironment = ClientEnvironment.PRODUCTION,
            client_name: typing.Optional[str] = None,
            timeout: typing.Optional[float] = 60,
            httpx_client: typing.Optional[httpx.Client] = None,
    ):
        if api_key is None:
            api_key = os.getenv("CO_API_KEY")

        BaseCohere.__init__(
            self,
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            token=api_key,
            timeout=timeout,
            httpx_client=httpx_client,
        )

        validate_args(self, "chat", throw_if_stream_is_true)

    wait = wait

    """
    The following methods have been moved or deprecated in cohere==5.0.0. Please update your usage.
    Issues may be filed in https://github.com/cohere-ai/cohere-python/issues.
    """
    check_api_key: Never = deprecated_function("check_api_key")
    loglikelihood: Never = deprecated_function("loglikelihood")
    batch_generate: Never = deprecated_function("batch_generate")
    codebook: Never = deprecated_function("codebook")
    batch_tokenize: Never = deprecated_function("batch_tokenize")
    batch_detokenize: Never = deprecated_function("batch_detokenize")
    detect_language: Never = deprecated_function("detect_language")
    generate_feedback: Never = deprecated_function("generate_feedback")
    generate_preference_feedback: Never = deprecated_function("generate_preference_feedback")
    create_dataset: Never = moved_function("create_dataset", ".datasets.create")
    get_dataset: Never = moved_function("get_dataset", ".datasets.get")
    list_datasets: Never = moved_function("list_datasets", ".datasets.list")
    delete_dataset: Never = moved_function("delete_dataset", ".datasets.delete")
    get_dataset_usage: Never = moved_function("get_dataset_usage", ".datasets.get_usage")
    wait_for_dataset: Never = moved_function("wait_for_dataset", ".wait")
    _check_response: Never = deprecated_function("_check_response")
    _request: Never = deprecated_function("_request")
    create_cluster_job: Never = deprecated_function("create_cluster_job")
    get_cluster_job: Never = deprecated_function("get_cluster_job")
    list_cluster_jobs: Never = deprecated_function("list_cluster_jobs")
    wait_for_cluster_job: Never = deprecated_function("wait_for_cluster_job")
    create_embed_job: Never = moved_function("create_embed_job", ".embed_jobs.create")
    list_embed_jobs: Never = moved_function("list_embed_jobs", ".embed_jobs.list")
    get_embed_job: Never = moved_function("get_embed_job", ".embed_jobs.get")
    cancel_embed_job: Never = moved_function("cancel_embed_job", ".embed_jobs.cancel")
    wait_for_embed_job: Never = moved_function("wait_for_embed_job", ".wait")
    create_custom_model: Never = deprecated_function("create_custom_model")
    wait_for_custom_model: Never = deprecated_function("wait_for_custom_model")
    _upload_dataset: Never = deprecated_function("_upload_dataset")
    _create_signed_url: Never = deprecated_function("_create_signed_url")
    get_custom_model: Never = deprecated_function("get_custom_model")
    get_custom_model_by_name: Never = deprecated_function("get_custom_model_by_name")
    get_custom_model_metrics: Never = deprecated_function("get_custom_model_metrics")
    list_custom_models: Never = deprecated_function("list_custom_models")
    create_connector: Never = moved_function("create_connector", ".connectors.create")
    update_connector: Never = moved_function("update_connector", ".connectors.update")
    get_connector: Never = moved_function("get_connector", ".connectors.get")
    list_connectors: Never = moved_function("list_connectors", ".connectors.list")
    delete_connector: Never = moved_function("delete_connector", ".connectors.delete")
    oauth_authorize_connector: Never = moved_function("oauth_authorize_connector", ".connectors.o_auth_authorize")


class AsyncClient(AsyncBaseCohere):
    def __init__(
            self,
            api_key: typing.Optional[typing.Union[str, typing.Callable[[], str]]] = None,
            *,
            base_url: typing.Optional[str] = os.getenv("CO_API_URL"),
            environment: ClientEnvironment = ClientEnvironment.PRODUCTION,
            client_name: typing.Optional[str] = None,
            timeout: typing.Optional[float] = 60,
            httpx_client: typing.Optional[httpx.AsyncClient] = None,
    ):
        if api_key is None:
            api_key = os.getenv("CO_API_KEY")

        AsyncBaseCohere.__init__(
            self,
            base_url=base_url,
            environment=environment,
            client_name=client_name,
            token=api_key,
            timeout=timeout,
            httpx_client=httpx_client,
        )

        validate_args(self, "chat", throw_if_stream_is_true)

    wait = async_wait

    """
    The following methods have been moved or deprecated in cohere==5.0.0. Please update your usage.
    Issues may be filed in https://github.com/cohere-ai/cohere-python/issues.
    """
    check_api_key: Never = deprecated_function("check_api_key")
    loglikelihood: Never = deprecated_function("loglikelihood")
    batch_generate: Never = deprecated_function("batch_generate")
    codebook: Never = deprecated_function("codebook")
    batch_tokenize: Never = deprecated_function("batch_tokenize")
    batch_detokenize: Never = deprecated_function("batch_detokenize")
    detect_language: Never = deprecated_function("detect_language")
    generate_feedback: Never = deprecated_function("generate_feedback")
    generate_preference_feedback: Never = deprecated_function("generate_preference_feedback")
    create_dataset: Never = moved_function("create_dataset", ".datasets.create")
    get_dataset: Never = moved_function("get_dataset", ".datasets.get")
    list_datasets: Never = moved_function("list_datasets", ".datasets.list")
    delete_dataset: Never = moved_function("delete_dataset", ".datasets.delete")
    get_dataset_usage: Never = moved_function("get_dataset_usage", ".datasets.get_usage")
    wait_for_dataset: Never = moved_function("wait_for_dataset", ".wait")
    _check_response: Never = deprecated_function("_check_response")
    _request: Never = deprecated_function("_request")
    create_cluster_job: Never = deprecated_function("create_cluster_job")
    get_cluster_job: Never = deprecated_function("get_cluster_job")
    list_cluster_jobs: Never = deprecated_function("list_cluster_jobs")
    wait_for_cluster_job: Never = deprecated_function("wait_for_cluster_job")
    create_embed_job: Never = moved_function("create_embed_job", ".embed_jobs.create")
    list_embed_jobs: Never = moved_function("list_embed_jobs", ".embed_jobs.list")
    get_embed_job: Never = moved_function("get_embed_job", ".embed_jobs.get")
    cancel_embed_job: Never = moved_function("cancel_embed_job", ".embed_jobs.cancel")
    wait_for_embed_job: Never = moved_function("wait_for_embed_job", ".wait")
    create_custom_model: Never = deprecated_function("create_custom_model")
    wait_for_custom_model: Never = deprecated_function("wait_for_custom_model")
    _upload_dataset: Never = deprecated_function("_upload_dataset")
    _create_signed_url: Never = deprecated_function("_create_signed_url")
    get_custom_model: Never = deprecated_function("get_custom_model")
    get_custom_model_by_name: Never = deprecated_function("get_custom_model_by_name")
    get_custom_model_metrics: Never = deprecated_function("get_custom_model_metrics")
    list_custom_models: Never = deprecated_function("list_custom_models")
    create_connector: Never = moved_function("create_connector", ".connectors.create")
    update_connector: Never = moved_function("update_connector", ".connectors.update")
    get_connector: Never = moved_function("get_connector", ".connectors.get")
    list_connectors: Never = moved_function("list_connectors", ".connectors.list")
    delete_connector: Never = moved_function("delete_connector", ".connectors.delete")
    oauth_authorize_connector: Never = moved_function("oauth_authorize_connector", ".connectors.o_auth_authorize")
