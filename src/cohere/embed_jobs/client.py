# This file was auto-generated by Fern from our API Definition.

import typing
import urllib.parse
from json.decoder import JSONDecodeError

from ..core.api_error import ApiError
from ..core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ..core.jsonable_encoder import jsonable_encoder
from ..core.query_encoder import encode_query
from ..core.remove_none_from_dict import remove_none_from_dict
from ..core.request_options import RequestOptions
from ..core.unchecked_base_model import construct_type
from ..errors.bad_request_error import BadRequestError
from ..errors.internal_server_error import InternalServerError
from ..errors.not_found_error import NotFoundError
from ..errors.too_many_requests_error import TooManyRequestsError
from ..types.create_embed_job_response import CreateEmbedJobResponse
from ..types.embed_input_type import EmbedInputType
from ..types.embed_job import EmbedJob
from ..types.embedding_type import EmbeddingType
from ..types.list_embed_job_response import ListEmbedJobResponse
from ..types.too_many_requests_error_body import TooManyRequestsErrorBody
from .types.create_embed_job_request_truncate import CreateEmbedJobRequestTruncate

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class EmbedJobsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> ListEmbedJobResponse:
        """
        The list embed job endpoint allows users to view all embed jobs history for that specific user.

        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        ListEmbedJobResponse
            OK

        Examples
        --------
        from cohere.client import Client

        client = Client(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        client.embed_jobs.list()
        """
        _response = self._client_wrapper.httpx_client.request(
            method="GET",
            url=urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", "embed-jobs"),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return typing.cast(ListEmbedJobResponse, construct_type(type_=ListEmbedJobResponse, object_=_response.json()))  # type: ignore
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def create(
        self,
        *,
        model: str,
        dataset_id: str,
        input_type: EmbedInputType,
        name: typing.Optional[str] = OMIT,
        embedding_types: typing.Optional[typing.Sequence[EmbeddingType]] = OMIT,
        truncate: typing.Optional[CreateEmbedJobRequestTruncate] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> CreateEmbedJobResponse:
        """
        This API launches an async Embed job for a [Dataset](https://docs.cohere.com/docs/datasets) of type `embed-input`. The result of a completed embed job is new Dataset of type `embed-output`, which contains the original text entries and the corresponding embeddings.

        Parameters
        ----------
        model : str
            ID of the embedding model.

            Available models and corresponding embedding dimensions:

            - `embed-english-v3.0` : 1024
            - `embed-multilingual-v3.0` : 1024
            - `embed-english-light-v3.0` : 384
            - `embed-multilingual-light-v3.0` : 384


        dataset_id : str
            ID of a [Dataset](https://docs.cohere.com/docs/datasets). The Dataset must be of type `embed-input` and must have a validation status `Validated`

        input_type : EmbedInputType

        name : typing.Optional[str]
            The name of the embed job.

        embedding_types : typing.Optional[typing.Sequence[EmbeddingType]]
            Specifies the types of embeddings you want to get back. Not required and default is None, which returns the Embed Floats response type. Can be one or more of the following types.

            * `"float"`: Use this when you want to get back the default float embeddings. Valid for all models.
            * `"int8"`: Use this when you want to get back signed int8 embeddings. Valid for only v3 models.
            * `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Valid for only v3 models.
            * `"binary"`: Use this when you want to get back signed binary embeddings. Valid for only v3 models.
            * `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Valid for only v3 models.

        truncate : typing.Optional[CreateEmbedJobRequestTruncate]
            One of `START|END` to specify how the API will handle inputs longer than the maximum token length.

            Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.


        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        CreateEmbedJobResponse
            OK

        Examples
        --------
        from cohere.client import Client

        client = Client(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        client.embed_jobs.create(
            model="model",
            dataset_id="dataset_id",
            input_type="search_document",
        )
        """
        _request: typing.Dict[str, typing.Any] = {"model": model, "dataset_id": dataset_id, "input_type": input_type}
        if name is not OMIT:
            _request["name"] = name
        if embedding_types is not OMIT:
            _request["embedding_types"] = embedding_types
        if truncate is not OMIT:
            _request["truncate"] = truncate
        _response = self._client_wrapper.httpx_client.request(
            method="POST",
            url=urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", "embed-jobs"),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            json=jsonable_encoder(_request)
            if request_options is None or request_options.get("additional_body_parameters") is None
            else {
                **jsonable_encoder(_request),
                **(jsonable_encoder(remove_none_from_dict(request_options.get("additional_body_parameters", {})))),
            },
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return typing.cast(CreateEmbedJobResponse, construct_type(type_=CreateEmbedJobResponse, object_=_response.json()))  # type: ignore
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def get(self, id: str, *, request_options: typing.Optional[RequestOptions] = None) -> EmbedJob:
        """
        This API retrieves the details about an embed job started by the same user.

        Parameters
        ----------
        id : str
            The ID of the embed job to retrieve.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        EmbedJob
            OK

        Examples
        --------
        from cohere.client import Client

        client = Client(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        client.embed_jobs.get(
            id="id",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            method="GET",
            url=urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", f"embed-jobs/{jsonable_encoder(id)}"),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return typing.cast(EmbedJob, construct_type(type_=EmbedJob, object_=_response.json()))  # type: ignore
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 404:
            raise NotFoundError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    def cancel(self, id: str, *, request_options: typing.Optional[RequestOptions] = None) -> None:
        """
        This API allows users to cancel an active embed job. Once invoked, the embedding process will be terminated, and users will be charged for the embeddings processed up to the cancellation point. It's important to note that partial results will not be available to users after cancellation.

        Parameters
        ----------
        id : str
            The ID of the embed job to cancel.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        None

        Examples
        --------
        from cohere.client import Client

        client = Client(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        client.embed_jobs.cancel(
            id="id",
        )
        """
        _response = self._client_wrapper.httpx_client.request(
            method="POST",
            url=urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", f"embed-jobs/{jsonable_encoder(id)}/cancel"
            ),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            json=jsonable_encoder(remove_none_from_dict(request_options.get("additional_body_parameters", {})))
            if request_options is not None
            else None,
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 404:
            raise NotFoundError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)


class AsyncEmbedJobsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> ListEmbedJobResponse:
        """
        The list embed job endpoint allows users to view all embed jobs history for that specific user.

        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        ListEmbedJobResponse
            OK

        Examples
        --------
        from cohere.client import AsyncClient

        client = AsyncClient(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        await client.embed_jobs.list()
        """
        _response = await self._client_wrapper.httpx_client.request(
            method="GET",
            url=urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", "embed-jobs"),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return typing.cast(ListEmbedJobResponse, construct_type(type_=ListEmbedJobResponse, object_=_response.json()))  # type: ignore
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def create(
        self,
        *,
        model: str,
        dataset_id: str,
        input_type: EmbedInputType,
        name: typing.Optional[str] = OMIT,
        embedding_types: typing.Optional[typing.Sequence[EmbeddingType]] = OMIT,
        truncate: typing.Optional[CreateEmbedJobRequestTruncate] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> CreateEmbedJobResponse:
        """
        This API launches an async Embed job for a [Dataset](https://docs.cohere.com/docs/datasets) of type `embed-input`. The result of a completed embed job is new Dataset of type `embed-output`, which contains the original text entries and the corresponding embeddings.

        Parameters
        ----------
        model : str
            ID of the embedding model.

            Available models and corresponding embedding dimensions:

            - `embed-english-v3.0` : 1024
            - `embed-multilingual-v3.0` : 1024
            - `embed-english-light-v3.0` : 384
            - `embed-multilingual-light-v3.0` : 384


        dataset_id : str
            ID of a [Dataset](https://docs.cohere.com/docs/datasets). The Dataset must be of type `embed-input` and must have a validation status `Validated`

        input_type : EmbedInputType

        name : typing.Optional[str]
            The name of the embed job.

        embedding_types : typing.Optional[typing.Sequence[EmbeddingType]]
            Specifies the types of embeddings you want to get back. Not required and default is None, which returns the Embed Floats response type. Can be one or more of the following types.

            * `"float"`: Use this when you want to get back the default float embeddings. Valid for all models.
            * `"int8"`: Use this when you want to get back signed int8 embeddings. Valid for only v3 models.
            * `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Valid for only v3 models.
            * `"binary"`: Use this when you want to get back signed binary embeddings. Valid for only v3 models.
            * `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Valid for only v3 models.

        truncate : typing.Optional[CreateEmbedJobRequestTruncate]
            One of `START|END` to specify how the API will handle inputs longer than the maximum token length.

            Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.


        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        CreateEmbedJobResponse
            OK

        Examples
        --------
        from cohere.client import AsyncClient

        client = AsyncClient(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        await client.embed_jobs.create(
            model="model",
            dataset_id="dataset_id",
            input_type="search_document",
        )
        """
        _request: typing.Dict[str, typing.Any] = {"model": model, "dataset_id": dataset_id, "input_type": input_type}
        if name is not OMIT:
            _request["name"] = name
        if embedding_types is not OMIT:
            _request["embedding_types"] = embedding_types
        if truncate is not OMIT:
            _request["truncate"] = truncate
        _response = await self._client_wrapper.httpx_client.request(
            method="POST",
            url=urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", "embed-jobs"),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            json=jsonable_encoder(_request)
            if request_options is None or request_options.get("additional_body_parameters") is None
            else {
                **jsonable_encoder(_request),
                **(jsonable_encoder(remove_none_from_dict(request_options.get("additional_body_parameters", {})))),
            },
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return typing.cast(CreateEmbedJobResponse, construct_type(type_=CreateEmbedJobResponse, object_=_response.json()))  # type: ignore
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def get(self, id: str, *, request_options: typing.Optional[RequestOptions] = None) -> EmbedJob:
        """
        This API retrieves the details about an embed job started by the same user.

        Parameters
        ----------
        id : str
            The ID of the embed job to retrieve.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        EmbedJob
            OK

        Examples
        --------
        from cohere.client import AsyncClient

        client = AsyncClient(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        await client.embed_jobs.get(
            id="id",
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            method="GET",
            url=urllib.parse.urljoin(f"{self._client_wrapper.get_base_url()}/", f"embed-jobs/{jsonable_encoder(id)}"),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return typing.cast(EmbedJob, construct_type(type_=EmbedJob, object_=_response.json()))  # type: ignore
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 404:
            raise NotFoundError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)

    async def cancel(self, id: str, *, request_options: typing.Optional[RequestOptions] = None) -> None:
        """
        This API allows users to cancel an active embed job. Once invoked, the embedding process will be terminated, and users will be charged for the embeddings processed up to the cancellation point. It's important to note that partial results will not be available to users after cancellation.

        Parameters
        ----------
        id : str
            The ID of the embed job to cancel.

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        None

        Examples
        --------
        from cohere.client import AsyncClient

        client = AsyncClient(
            client_name="YOUR_CLIENT_NAME",
            token="YOUR_TOKEN",
        )
        await client.embed_jobs.cancel(
            id="id",
        )
        """
        _response = await self._client_wrapper.httpx_client.request(
            method="POST",
            url=urllib.parse.urljoin(
                f"{self._client_wrapper.get_base_url()}/", f"embed-jobs/{jsonable_encoder(id)}/cancel"
            ),
            params=encode_query(
                jsonable_encoder(
                    request_options.get("additional_query_parameters") if request_options is not None else None
                )
            ),
            json=jsonable_encoder(remove_none_from_dict(request_options.get("additional_body_parameters", {})))
            if request_options is not None
            else None,
            headers=jsonable_encoder(
                remove_none_from_dict(
                    {
                        **self._client_wrapper.get_headers(),
                        **(request_options.get("additional_headers", {}) if request_options is not None else {}),
                    }
                )
            ),
            timeout=request_options.get("timeout_in_seconds")
            if request_options is not None and request_options.get("timeout_in_seconds") is not None
            else self._client_wrapper.get_timeout(),
            retries=0,
            max_retries=request_options.get("max_retries") if request_options is not None else 0,  # type: ignore
        )
        if 200 <= _response.status_code < 300:
            return
        if _response.status_code == 400:
            raise BadRequestError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 404:
            raise NotFoundError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 429:
            raise TooManyRequestsError(
                typing.cast(TooManyRequestsErrorBody, construct_type(type_=TooManyRequestsErrorBody, object_=_response.json()))  # type: ignore
            )
        if _response.status_code == 500:
            raise InternalServerError(
                typing.cast(typing.Any, construct_type(type_=typing.Any, object_=_response.json()))  # type: ignore
            )
        try:
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, body=_response.text)
        raise ApiError(status_code=_response.status_code, body=_response_json)
