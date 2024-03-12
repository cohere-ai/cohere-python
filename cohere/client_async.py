import asyncio
import json as jsonlib
import os
import posixpath
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from typing import Any, BinaryIO, Callable, Dict, Iterable, List, Optional, Union

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

import aiohttp
import backoff

import cohere
from cohere.client import Client
from cohere.custom_model_dataset import CustomModelDataset
from cohere.error import CohereAPIError, CohereConnectionError, CohereError
from cohere.logging import logger
from cohere.responses import (
    Classification,
    Classifications,
    ClusterJobResult,
    Codebook,
    Connector,
    DetectLanguageResponse,
    Detokenization,
    EmbeddingResponses,
    Embeddings,
    GenerateFeedbackResponse,
    GeneratePreferenceFeedbackResponse,
    Generations,
    LabelPrediction,
    LogLikelihoods,
    PreferenceRating,
    Reranking,
    StreamingGenerations,
    SummarizeResponse,
    Tokens,
)
from cohere.responses.chat import (
    AsyncChat,
    ChatRequestToolResultsItem,
    StreamingChat,
    Tool,
)
from cohere.responses.classify import Example as ClassifyExample
from cohere.responses.cluster import AsyncClusterJobResult
from cohere.responses.custom_model import (
    CUSTOM_MODEL_INTERNAL_STATUS_MAPPING,
    CUSTOM_MODEL_PRODUCT_MAPPING,
    CUSTOM_MODEL_STATUS,
    CUSTOM_MODEL_TYPE,
    INTERNAL_CUSTOM_MODEL_TYPE,
    AsyncCustomModel,
    HyperParametersInput,
    ModelMetric,
)
from cohere.responses.dataset import AsyncDataset, Dataset, DatasetUsage, ParseInfo
from cohere.responses.embed_job import AsyncEmbedJob
from cohere.utils import async_wait_for_job, is_api_key_valid, np_json_dumps

JSON = Union[Dict, List]


class AsyncClient(Client):
    """AsyncClient

    This client provides an asyncio/aiohttp interface.
    Using this client is recommended when you are making highly parallel request,
    or when calling the Cohere API from a server such as FastAPI."""

    def __init__(
        self,
        api_key: str = None,
        num_workers: int = 16,
        request_dict: dict = {},
        check_api_key: bool = True,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout=300,
        api_url: str = None,
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = api_url or os.getenv("CO_API_URL", cohere.COHERE_API_URL)
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk-" + cohere.SDK_VERSION
        self.max_retries = max_retries
        if client_name:
            self.request_source += ":" + client_name
        self.api_version = f"v{cohere.API_VERSION}"
        self._check_api_key_on_enter = check_api_key
        self._backend = AIOHTTPBackend(logger, num_workers, max_retries, timeout)

    async def _request(
        self, endpoint, json=None, files=None, method="POST", full_url=None, stream=False, params=None
    ) -> JSON:
        headers = {
            "Authorization": f"BEARER {self.api_key}",
            "Request-Source": self.request_source,
        }
        if json:
            headers["Content-Type"] = "application/json"

        if endpoint is None and full_url is not None:  # api key
            url = full_url
        else:
            url = posixpath.join(self.api_url, self.api_version, endpoint)

        response = await self._backend.request(url, json, files, method, headers, stream=stream, params=params)
        if stream:
            return response

        try:
            json_response = await response.json()
        #   `CohereAPIError.from_aio_response()` will capture the http status code
        except jsonlib.decoder.JSONDecodeError:
            raise CohereAPIError.from_aio_response(
                response, message=f"Failed to decode json body: {await response.text()}"
            )
        except aiohttp.ClientPayloadError as e:
            raise CohereAPIError.from_aio_response(
                response, message=f"An unexpected error occurred while receiving the response: {e}"
            )

        logger.debug(f"JSON response: {json_response}")
        self._check_response(json_response, response.headers, response.status)
        return json_response

    async def close(self):
        return await self._backend.close()

    async def __aenter__(self):
        if self._check_api_key_on_enter:
            await self.check_api_key()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    # API methods
    async def check_api_key(self) -> Dict[str, bool]:
        """
        check_api_key raises an exception when the key is invalid, but the return value for valid keys is kept for
        backwards compatibility.
        """
        return {"valid": is_api_key_valid(self.api_key)}

    async def loglikelihood(
        self,
        prompt: Optional[str] = None,
        completion: Optional[str] = None,
        model: Optional[str] = None,
    ) -> LogLikelihoods:
        json_body = {"model": model, "prompt": prompt, "completion": completion}
        response = await self._request(cohere.LOGLIKELIHOOD_URL, json=json_body)
        return LogLikelihoods(response["prompt_tokens"], response["completion_tokens"])

    async def batch_generate(
        self, prompts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[Exception, Generations]]:
        return await asyncio.gather(
            *[self.generate(prompt, **kwargs) for prompt in prompts], return_exceptions=return_exceptions
        )

    async def generate(
        self,
        prompt: Optional[str] = None,
        prompt_vars: object = {},
        model: Optional[str] = None,
        preset: Optional[str] = None,
        num_generations: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        end_sequences: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,
        return_likelihoods: Optional[str] = None,
        truncate: Optional[str] = None,
        stream: bool = False,
    ) -> Union[Generations, StreamingGenerations]:
        json_body = {
            "model": model,
            "prompt": prompt,
            "prompt_vars": prompt_vars,
            "preset": preset,
            "num_generations": num_generations,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": k,
            "p": p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "end_sequences": end_sequences,
            "stop_sequences": stop_sequences,
            "return_likelihoods": return_likelihoods,
            "truncate": truncate,
            "stream": stream,
        }
        response = await self._request(cohere.GENERATE_URL, json=json_body, stream=stream)
        if stream:
            return StreamingGenerations(response)
        else:
            return Generations.from_dict(response=response, return_likelihoods=return_likelihoods)

    async def chat(
        self,
        message: Optional[str] = None,
        conversation_id: Optional[str] = "",
        model: Optional[str] = None,
        return_chat_history: Optional[bool] = False,
        return_prompt: Optional[bool] = False,
        return_preamble: Optional[bool] = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
        preamble: Optional[str] = None,
        user_name: Optional[str] = None,
        temperature: Optional[float] = 0.8,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        p: Optional[float] = None,
        k: Optional[float] = None,
        search_queries_only: Optional[bool] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        citation_quality: Optional[str] = None,
        prompt_truncation: Optional[str] = None,
        connectors: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Tool]] = None,
        tool_results: Optional[List[ChatRequestToolResultsItem]] = None,
        raw_prompting: Optional[bool] = False,
    ) -> Union[AsyncChat, StreamingChat]:
        if message is None:
            raise CohereError("'message' must be provided.")

        json_body = {
            "message": message,
            "conversation_id": conversation_id,
            "model": model,
            "return_chat_history": return_chat_history,
            "return_prompt": return_prompt,
            "return_preamble": return_preamble,
            "chat_history": chat_history,
            "preamble": preamble,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "user_name": user_name,
            "p": p,
            "k": k,
            "search_queries_only": search_queries_only,
            "documents": documents,
            "connectors": connectors,
            "tools": tools,
            "tool_results": tool_results,
            "raw_prompting": raw_prompting,
        }
        if citation_quality is not None:
            json_body["citation_quality"] = citation_quality
        if prompt_truncation is not None:
            json_body["prompt_truncation"] = prompt_truncation

        response = await self._request(cohere.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return AsyncChat.from_dict(response, message=message, client=self)

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        truncate: Optional[str] = None,
        input_type: Optional[str] = None,
        embedding_types: Optional[List[str]] = None,
    ) -> Embeddings:
        """Returns an Embeddings object for the provided texts. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            text (List[str]): A list of strings to embed.
            model (str): (Optional) The model ID to use for embedding the text.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
            input_type (str): (Optional) One of "classification", "clustering", "search_document", "search_query". The type of input text provided to embed.
            embedding_types (List[str]): (Optional) Specifies the types of embeddings you want to get back. Not required and default is None, which returns the float embeddings in the response's embeddings field. Can be one or more of the following types: "float", "int8", "uint8", "binary", "ubinary".
        """
        json_bodys = [
            dict(
                texts=texts[i : i + cohere.COHERE_EMBED_BATCH_SIZE],
                model=model,
                truncate=truncate,
                input_type=input_type,
                embedding_types=embedding_types,
            )
            for i in range(0, len(texts), cohere.COHERE_EMBED_BATCH_SIZE)
        ]
        responses = await asyncio.gather(*[self._request(cohere.EMBED_URL, json) for json in json_bodys])
        meta = responses[0]["meta"] if responses else None
        embedding_responses = EmbeddingResponses()
        for response in responses:
            embedding_responses.add_response(response)

        return Embeddings(
            embeddings=embedding_responses.get_embeddings(),
            response_type=embedding_responses.response_type,
            meta=meta,
        )

    async def codebook(
        self,
        model: Optional[str] = None,
        compression_codebook: Optional[str] = "default",
    ) -> Codebook:
        """Returns a codebook object for the provided model. Visit https://cohere.ai/embed to learn about compressed embeddings and codebooks.

        Args:
            model (str): (Optional) The model ID to use for embedding the text.
            compression_codebook (str): (Optional) The compression codebook to use for compressed embeddings. Defaults to "default".
        """
        json_body = {
            "model": model,
            "compression_codebook": compression_codebook,
        }
        response = await self._request(cohere.CODEBOOK_URL, json=json_body)
        return Codebook(response["codebook"], response["meta"])

    async def classify(
        self,
        inputs: List[str] = [],
        model: Optional[str] = None,
        preset: Optional[str] = None,
        examples: List[ClassifyExample] = [],
        truncate: Optional[str] = None,
    ) -> Classifications:
        examples_dicts = [{"text": example.text, "label": example.label} for example in examples]

        json_body = {
            "model": model,
            "preset": preset,
            "inputs": inputs,
            "examples": examples_dicts,
            "truncate": truncate,
        }
        response = await self._request(cohere.CLASSIFY_URL, json=json_body)
        classifications = []
        for res in response["classifications"]:
            labelObj = {}
            for label, prediction in res["labels"].items():
                labelObj[label] = LabelPrediction(prediction["confidence"])
            classifications.append(
                Classification(
                    input=res["input"],
                    predictions=res.get("predictions", None),
                    confidences=res.get("confidences", None),
                    prediction=res.get("prediction", None),
                    confidence=res.get("confidence", None),
                    labels=labelObj,
                    classification_type=res.get("classification_type", "single-label"),
                    id=res["id"],
                )
            )

        return Classifications(classifications, response["meta"])

    async def summarize(
        self,
        text: str,
        model: Optional[str] = None,
        length: Optional[str] = None,
        format: Optional[str] = None,
        temperature: Optional[float] = None,
        additional_command: Optional[str] = None,
        extractiveness: Optional[str] = None,
    ) -> SummarizeResponse:
        json_body = {
            "model": model,
            "text": text,
            "length": length,
            "format": format,
            "temperature": temperature,
            "additional_command": additional_command,
            "extractiveness": extractiveness,
        }
        # remove None values from the dict
        json_body = {k: v for k, v in json_body.items() if v is not None}
        response = await self._request(cohere.SUMMARIZE_URL, json=json_body)
        return SummarizeResponse(id=response["id"], summary=response["summary"], meta=response["meta"])

    async def batch_tokenize(
        self, texts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[Tokens, Exception]]:
        return await asyncio.gather(*[self.tokenize(t, **kwargs) for t in texts], return_exceptions=return_exceptions)

    async def tokenize(self, text: str, model: Optional[str] = None) -> Tokens:
        json_body = {"text": text, "model": model}
        res = await self._request(cohere.TOKENIZE_URL, json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"], meta=res["meta"])

    async def batch_detokenize(
        self, list_of_tokens: List[List[int]], return_exceptions=False, **kwargs
    ) -> List[Union[Detokenization, Exception]]:
        return await asyncio.gather(
            *[self.detokenize(t, **kwargs) for t in list_of_tokens], return_exceptions=return_exceptions
        )

    async def detokenize(self, tokens: List[int], model: Optional[str] = None) -> Detokenization:
        json_body = {"tokens": tokens, "model": model}
        res = await self._request(cohere.DETOKENIZE_URL, json_body)
        return Detokenization(text=res["text"], meta=res["meta"])

    async def detect_language(self, texts: List[str]) -> DetectLanguageResponse:
        """
        This API is deprecated.
        """
        raise DeprecationWarning("The detect_language API is no longer supported")

    async def generate_feedback(
        self,
        request_id: str,
        good_response: bool,
        model=None,
        desired_response: str = None,
        flagged_response: bool = None,
        flagged_reason: str = None,
        prompt: str = None,
        annotator_id: str = None,
    ) -> GenerateFeedbackResponse:
        json_body = {
            "request_id": request_id,
            "good_response": good_response,
            "desired_response": desired_response,
            "flagged_response": flagged_response,
            "flagged_reason": flagged_reason,
            "prompt": prompt,
            "annotator_id": annotator_id,
            "model": model,
        }
        response = await self._request(cohere.GENERATE_FEEDBACK_URL, json_body)
        return GenerateFeedbackResponse(id=response["id"])

    async def generate_preference_feedback(
        self,
        ratings: List[PreferenceRating],
        model=None,
        prompt: str = None,
        annotator_id: str = None,
    ) -> GeneratePreferenceFeedbackResponse:
        ratings_dicts = []
        for rating in ratings:
            ratings_dicts.append(asdict(rating))
        json_body = {
            "ratings": ratings_dicts,
            "prompt": prompt,
            "annotator_id": annotator_id,
            "model": model,
        }
        response = await self._request(cohere.GENERATE_PREFERENCE_FEEDBACK_URL, json_body)
        return GeneratePreferenceFeedbackResponse(id=response["id"])

    async def rerank(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        model: str,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
    ) -> Reranking:
        """Returns an ordered list of documents ordered by their relevance to the provided query

        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            model (str): The model to use for re-ranking
            top_n (int): (optional) The number of results to return, defaults to returning all results
            max_chunks_per_doc (int): (optional) The maximum number of chunks derived from a document
        """
        parsed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                parsed_docs.append({"text": doc})
            elif isinstance(doc, dict) and "text" in doc:
                parsed_docs.append(doc)
            else:
                raise CohereError(
                    message='invalid format for documents, must be a list of strings or dicts with a "text" key'
                )

        json_body = {
            "query": query,
            "documents": parsed_docs,
            "model": model,
            "top_n": top_n,
            "return_documents": False,
            "max_chunks_per_doc": max_chunks_per_doc,
        }
        reranking = Reranking(await self._request(cohere.RERANK_URL, json=json_body))
        for rank in reranking.results:
            rank.document = parsed_docs[rank.index]
        return reranking

    async def create_dataset(
        self,
        name: str,
        data: BinaryIO,
        dataset_type: str,
        eval_data: Optional[BinaryIO] = None,
        keep_fields: Union[str, List[str]] = None,
        optional_fields: Union[str, List[str]] = None,
        parse_info: Optional[ParseInfo] = None,
    ) -> AsyncDataset:
        """Returns a Dataset given input data

        Args:
            name (str): The name of your dataset
            data (BinaryIO): The data to be uploaded and validated
            dataset_type (str): The type of dataset you want to upload
            eval_data (BinaryIO): (optional) If the dataset type supports it upload evaluation data
            keep_fields (Union[str, List[str]]): (optional) A list of fields you want to keep in the dataset that are required
            optional_fields (Union[str, List[str]]): (optional) A list of fields you want to keep in the dataset that are optional
            parse_info: ParseInfo: (optional) information on how to parse the raw data
        Returns:
            AsyncDataset: Dataset object.
        """
        files = {"data": data}
        if eval_data:
            files["eval_data"] = eval_data
        params = {
            "name": name,
            "type": dataset_type,
        }
        if keep_fields:
            params["keep_fields"] = keep_fields
        if optional_fields:
            params["optional_fields"] = optional_fields
        if parse_info:
            params.update(parse_info.get_params())

        logger.warning("uploading file, starting validation...")
        create_response = await self._request(cohere.DATASET_URL, files=files, params=params)
        logger.warning(f"{create_response['id']} was uploaded")
        return await self.get_dataset(id=create_response["id"])

    async def get_dataset(self, id: str) -> AsyncDataset:
        """Returns a Dataset given a dataset id

        Args:
            id (str): The name of id of your dataset

        Returns:
            AsyncDataset: Dataset object.
        """
        if not id:
            raise CohereError(message="id must not be empty")
        response = await self._request(f"{cohere.DATASET_URL}/{id}", method="GET")
        return AsyncDataset.from_dict(response["dataset"], wait_fn=self.wait_for_dataset)

    async def list_datasets(
        self, dataset_type: str = None, limit: int = None, offset: int = None
    ) -> List[AsyncDataset]:
        """Returns a list of your Datasets

        Args:
            dataset_type (str): (optional) The dataset_type to filter on
            limit (int): (optional) The max number of datasets to return
            offset (int): (optional) The number of datasets to offset by

        Returns:
            List[AsyncDataset]: List of Dataset objects.
        """
        param_dict = {}
        if dataset_type:
            param_dict["dataset_type"] = dataset_type
        if limit:
            param_dict["limit"] = limit
        if offset:
            param_dict["offset"] = offset

        response = await self._request(f"{cohere.DATASET_URL}", method="GET", params=param_dict)
        return [
            AsyncDataset.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_dataset)
            for r in (response.get("datasets") or [])
        ]

    async def delete_dataset(self, id: str) -> None:
        """Deletes your dataset

        Args:
            id (str): The id of the dataset to delete
        """
        self._request(f"{cohere.DATASET_URL}/{id}", method="DELETE")

    async def get_dataset_usage(self) -> DatasetUsage:
        """Gets your total storage used in datasets

        Returns:
            DatasetUsage: Object containg current dataset usage
        """
        response = self._request(f"{cohere.DATASET_URL}/usage", method="GET")
        return DatasetUsage.from_dict(response)

    async def wait_for_dataset(
        self,
        dataset_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> AsyncDataset:
        """Wait for Dataset validation result.

        Args:
            dataset_id (str): Dataset id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            AsyncDataset: Dataset object.
        """

        return async_wait_for_job(
            get_job=partial(self.get_dataset, dataset_id),
            timeout=timeout,
            interval=interval,
        )

    async def create_cluster_job(
        self,
        input_dataset_id: str = None,
        embeddings_url: str = None,
        min_cluster_size: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        is_deterministic: Optional[bool] = None,
        generate_descriptions: Optional[bool] = None,
    ) -> AsyncClusterJobResult:
        """Create clustering job.

        Args:
            input_dataset_id (str): Id of the dataset to cluster.
            embeddings_url (str): File with embeddings to cluster.
            min_cluster_size (Optional[int], optional): Minimum number of elements in a cluster. Defaults to 10.
            n_neighbors (Optional[int], optional): Number of nearest neighbors used by UMAP to establish the
                local structure of the data. Defaults to 15. For more information, please refer to
                https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
            is_deterministic (Optional[bool], optional): Determines whether the output of the cluster job is
                deterministic. Defaults to True.
            generate_descriptions (Optional[bool], optional): Determines whether to generate cluster descriptions. Defaults to False.

        Returns:
            AsyncClusterJobResult: Created clustering job
        """

        json_body = {
            "input_dataset_id": input_dataset_id,
            "embeddings_url": embeddings_url,
            "min_cluster_size": min_cluster_size,
            "n_neighbors": n_neighbors,
            "is_deterministic": is_deterministic,
            "generate_descriptions": generate_descriptions,
        }
        response = await self._request(cohere.CLUSTER_JOBS_URL, json=json_body)
        cluster_job = await self.get_cluster_job(response.get("job_id"))
        return cluster_job

    async def get_cluster_job(
        self,
        job_id: str,
    ) -> ClusterJobResult:
        """Get clustering job results.

        Args:
            job_id (str): Clustering job id.

        Raises:
            ValueError: "job_id" is empty

        Returns:
            ClusterJobResult: Clustering job result.
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        response = await self._request(f"{cohere.CLUSTER_JOBS_URL}/{job_id}", method="GET")
        return ClusterJobResult.from_dict(response, wait_fn=self.wait_for_cluster_job)

    async def list_cluster_jobs(self) -> List[ClusterJobResult]:
        """List clustering jobs.

        Returns:
            List[ClusterJobResult]: Clustering jobs created.
        """

        response = await self._request(cohere.CLUSTER_JOBS_URL, method="GET")
        return [
            ClusterJobResult.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_cluster_job)
            for r in response["jobs"]
        ]

    async def wait_for_cluster_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> ClusterJobResult:
        """Wait for clustering job result.

        Args:
            job_id (str): Clustering job id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            ClusterJobResult: Clustering job result.
        """

        return await async_wait_for_job(
            get_job=partial(self.get_cluster_job, job_id),
            timeout=timeout,
            interval=interval,
        )

    async def create_embed_job(
        self,
        dataset_id: str,
        model: str,
        input_type: str,
        name: Optional[str] = None,
        truncate: Optional[str] = None,
        embedding_types: Optional[List[str]] = None,
    ) -> AsyncEmbedJob:
        """Create embed job.

        Args:
            dataset_id (str): dataset id with text to embed.
            model (str): The model ID to use for embedding the text.
            input_type (str): One of "classification", "clustering", "search_document", "search_query". The type of input text provided to embed.
            truncate (Optional[str], optional): How the API handles text longer than the maximum token length. Defaults to None.
            name (Optional[str], optional): The name of the embed job. Defaults to None.
            embedding_types (List[str]): (Optional) Specifies the types of embeddings you want to get back. Not required and default is None, which returns the float embeddings in the response's embeddings field. Can be one or more of the following types: "float", "int8", "uint8", "binary", "ubinary".

        Returns:
            AsyncEmbedJob: The created embed job
        """

        json_body = {
            "dataset_id": dataset_id,
            "name": name,
            "model": model,
            "truncate": truncate,
            "input_type": input_type,
            "embedding_types": embedding_types,
        }

        response = await self._request(cohere.EMBED_JOBS_URL, json=json_body)
        embed_job = await self.get_embed_job(response.get("job_id"))

        return embed_job

    async def list_embed_jobs(self) -> List[AsyncEmbedJob]:
        """List embed jobs.

        Returns:
            List[AsyncEmbedJob]: embed jobs.
        """

        response = await self._request(f"{cohere.EMBED_JOBS_URL}", method="GET")
        return [
            AsyncEmbedJob.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_embed_job)
            for r in response["embed_jobs"]
        ]

    async def get_embed_job(self, job_id: str) -> AsyncEmbedJob:
        """Get embed job.

        Args:
            job_id (str): embed job id.

        Raises:
            ValueError: "job_id" is empty

        Returns:
            AsyncEmbedJob: embed job.
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        response = await self._request(f"{cohere.EMBED_JOBS_URL}/{job_id}", method="GET")
        job = AsyncEmbedJob.from_dict(response, wait_fn=self.wait_for_embed_job)
        if response.get("output_dataset_id"):
            job.output = self.get_dataset(response.get("output_dataset_id"))
        return job

    async def cancel_embed_job(self, job_id: str) -> None:
        """Cancel embed job.

        Args:
            job_id (str): embed job id.

        Raises:
            ValueError: "job_id" is empty
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        await self._request(f"{cohere.EMBED_JOBS_URL}/{job_id}/cancel", method="POST", json={})

    async def wait_for_embed_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> AsyncEmbedJob:
        """Wait for embed job completion.

        Args:
            job_id (str): embed job id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            AsyncEmbedJob: embed job.
        """

        return await async_wait_for_job(
            get_job=partial(self.get_embed_job, job_id),
            timeout=timeout,
            interval=interval,
        )

    async def create_custom_model(
        self,
        name: str,
        model_type: CUSTOM_MODEL_TYPE,
        dataset: Union[Dataset, str],
        base_model: Optional[str] = None,
        hyperparameters: Optional[HyperParametersInput] = None,
    ) -> AsyncCustomModel:
        """Create a new custom model

        Args:
            name (str): name of your custom model, has to be unique across your organization
            model_type (GENERATIVE, CLASSIFY, RERANK): type of custom model
            dataset (Dataset, str): A dataset or dataset id for your training.
            base_model (str): base model to use for your custom model.
                For generative and classify models, `base_model` has to be None (no option available for now)
                For rerank models, you can choose between `english` and `multilingual`. Defaults to `english` if not specified.
                    The English model is better for English, while the multilingual model should be picked if a non-negligible part of queries/documents
                    will be in other languages
            hyperparameters (HyperParametersInput): adjust hyperparameters for your custom model. Only for generative custom models.
        Returns:
            CustomModel: the custom model that was created

        Examples:
             prompt completion custom model with dataset
                >>> co = cohere.Client("YOUR_API_KEY")
                >>> ds = co.create_dataset(name="prompt-completion-datset", data=open("/path/to/your/file.csv", "rb"), dataset_type="prompt-completion-finetune-input")
                >>> ds.await_validation()
                >>> co.create_custom_model("prompt-completion-ft", model_type="GENERATIVE", train_dataset=ds.id)

             classification custom model with train and evaluation data
                >>> co = cohere.Client("YOUR_API_KEY")
                >>> ds = co.create_dataset(name="classify-datset", data=open("train_file.csv", "rb"), eval_data=open("eval_file", "rb"), dataset_type="single-label-classification-finetune-input")
                >>> ds.await_validation()
                >>> co.create_custom_model("classify-ft", model_type="CLASSIFY", train_dataset=ds.id)
        """
        internal_custom_model_type = CUSTOM_MODEL_PRODUCT_MAPPING[model_type]

        json = {
            "name": name,
            "settings": {
                "trainFiles": [],
                "evalFiles": [],
                "baseModel": base_model,
                "finetuneType": internal_custom_model_type,
            },
        }
        if hyperparameters:
            json["settings"]["hyperparameters"] = {
                "earlyStoppingPatience": hyperparameters.get("early_stopping_patience"),
                "earlyStoppingThreshold": hyperparameters.get("early_stopping_threshold"),
                "trainBatchSize": hyperparameters.get("train_batch_size"),
                "trainEpochs": hyperparameters.get("train_epochs"),
                "learningRate": hyperparameters.get("learning_rate"),
            }

        if isinstance(dataset, Dataset):
            if not dataset.has_terminal_status():
                dataset.wait()
            json["settings"]["datasetID"] = dataset.id
        elif isinstance(dataset, str):
            dataset = await self.get_dataset(dataset)
            if not dataset.has_terminal_status():
                await dataset.wait()
            json["settings"]["datasetID"] = dataset.id
        elif isinstance(dataset, CustomModelDataset):
            logger.warning("`CustomModelDataset` is deprecated, use `Dataset` instead.")
            remote_path = self._upload_dataset(
                dataset.get_train_data(), name, dataset.train_file_name(), internal_custom_model_type
            )
            json["settings"]["trainFiles"].append({"path": remote_path, **dataset.file_config()})
            if dataset.has_eval_file():
                remote_path = self._upload_dataset(
                    dataset.get_eval_data(), name, dataset.eval_file_name(), internal_custom_model_type
                )
                json["settings"]["evalFiles"].append({"path": remote_path, **dataset.file_config()})
        else:
            raise CohereError(f"unsupported type for dataset {type(dataset)}")

        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/CreateFinetune", method="POST", json=json)
        return AsyncCustomModel.from_dict(response["finetune"], self.wait_for_custom_model)

    async def wait_for_custom_model(
        self,
        custom_model_id: str,
        timeout: Optional[float] = None,
        interval: float = 60,
    ) -> AsyncCustomModel:
        """Wait for custom model training completion.

        Args:
            custom_model_id (str): Custom model id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            AsyncCustomModel: Custom model.
        """

        return await async_wait_for_job(
            get_job=partial(self.get_custom_model, custom_model_id),
            timeout=timeout,
            interval=interval,
        )

    async def _upload_dataset(
        self, content: Iterable[bytes], custom_model_name: str, file_name: str, type: INTERNAL_CUSTOM_MODEL_TYPE
    ) -> str:
        gcs = await self._create_signed_url(custom_model_name, file_name, type)
        session = await self._backend.session()
        response = await session.put(url=gcs["url"], data=b"".join(content), headers={"content-type": "text/plain"})
        if response.status != 200:
            raise CohereError(message=f"Unexpected server error (status {response.status}): {response.text}")
        return gcs["gcspath"]

    async def _create_signed_url(
        self, custom_model_name: str, file_name: str, type: INTERNAL_CUSTOM_MODEL_TYPE
    ) -> TypedDict("gcsData", {"url": str, "gcspath": str}):
        json = {"finetuneName": custom_model_name, "fileName": file_name, "finetuneType": type}
        return await self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneUploadSignedURL", method="POST", json=json)

    async def get_custom_model(self, custom_model_id: str) -> AsyncCustomModel:
        """Get a custom model by id.

        Args:
            custom_model_id (str): custom model id
        Returns:
            CustomModel: the custom model
        """
        json = {"finetuneID": custom_model_id}
        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetune", method="POST", json=json)
        return AsyncCustomModel.from_dict(response["finetune"], self.wait_for_custom_model)

    async def get_custom_model_by_name(self, name: str) -> AsyncCustomModel:
        """Get a custom model by name.

        Args:
            name (str): custom model name
        Returns:
            CustomModel: the custom model
        """
        json = {"name": name}
        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneByName", method="POST", json=json)
        return AsyncCustomModel.from_dict(response["finetune"], self.wait_for_custom_model)

    async def get_custom_model_metrics(self, custom_model_id: str) -> List[ModelMetric]:
        """Get model metrics by id

        Args:
            custom_model_id (str): custom model id
        Returns:
            List[ModelMetric]: a list of model metrics
        """
        json = {"finetuneID": custom_model_id}
        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneMetrics", method="POST", json=json)
        return [ModelMetric.from_dict(metric) for metric in response["metrics"]]

    async def list_custom_models(
        self,
        statuses: Optional[List[CUSTOM_MODEL_STATUS]] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        order_by: Optional[Literal["asc", "desc"]] = None,
    ) -> List[AsyncCustomModel]:
        """List custom models of your organization.

        Args:
            statuses (CUSTOM_MODEL_STATUS, optional): search for finetunes which are in one of these states
            before (datetime, optional): search for custom models that were created before this timestamp
            after (datetime, optional): search for custom models that were created after this timestamp
            order_by (Literal["asc", "desc"], optional): sort custom models by created at, either asc or desc
        Returns:
            List[CustomModel]: a list of custom models.
        """
        if before:
            before = before.replace(tzinfo=before.tzinfo or timezone.utc)
        if after:
            after = after.replace(tzinfo=after.tzinfo or timezone.utc)
        internal_statuses = []
        if statuses:
            for status in statuses:
                internal_statuses.append(CUSTOM_MODEL_INTERNAL_STATUS_MAPPING[status])
        json = {
            "query": {
                "statuses": internal_statuses,
                "before": before.isoformat(timespec="seconds") if before else None,
                "after": after.isoformat(timespec="seconds") if after else None,
                "orderBy": order_by,
            }
        }

        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/ListFinetunes", method="POST", json=json)
        return [AsyncCustomModel.from_dict(r, self.wait_for_custom_model) for r in response["finetunes"]]

    async def create_connector(
        self,
        name: str,
        url: str,
        active: bool = True,
        continue_on_failure: bool = False,
        excludes: Optional[List[str]] = None,
        oauth: Optional[dict] = None,
        service_auth: Optional[dict] = None,
    ) -> Connector:
        """Creates a Connector with the provided information

        Args:
            name (str): The name of your connector
            url (str): The URL of the connector that will be used to search for documents
            active (bool): (optional) Whether the connector is active or not
            continue_on_failure (bool): (optional) Whether a chat request should continue or not if the request to this connector fails
            excludes (List[str]): (optional) A list of fields to exclude from the prompt (fields remain in the document)
            oauth (dict): (optional) The OAuth 2.0 configuration for the connector.
            service_auth: (dict): (optional) The service to service authentication configuration for the connector
        Returns:
            Connector: Connector object.
        """
        json = {
            "name": name,
            "url": url,
            "active": active,
            "continue_on_failure": continue_on_failure,
        }
        if oauth is not None:
            json["oauth"] = oauth

        if service_auth is not None:
            json["service_auth"] = service_auth

        if excludes is not None:
            json["excludes"] = excludes

        create_response = await self._request(cohere.CONNECTOR_URL, json=json)
        return await self.get_connector(id=create_response["connector"]["id"])

    async def update_connector(
        self,
        id: str,
        name: Optional[str] = None,
        url: Optional[str] = None,
        active: Optional[bool] = None,
        continue_on_failure: Optional[bool] = None,
        excludes: Optional[List[str]] = None,
        oauth: Optional[dict] = None,
        service_auth: Optional[dict] = None,
    ) -> Connector:
        """Updates a Connector with the provided id

        Args:
            id (str): The ID of the connector you wish to update.
            name (str): (optional) The name of your connector
            url (str): (optional) The URL of the connector that will be used to search for documents
            active (bool): (optional) Whether the connector is active or not
            continue_on_failure (bool): (optional) Whether a chat request should continue or not if the request to this connector fails
            excludes (List[str]): (optional) A list of fields to exclude from the prompt (fields remain in the document)
            oauth (dict): (optional) The OAuth 2.0 configuration for the connector.
            service_auth: (dict): (optional) The service to service authentication configuration for the connector
        Returns:
            Connector: Connector object.
        """
        if not id:
            raise CohereError(message="id must not be empty")
        json = {}
        if url is not None:
            json["url"] = url

        if active is not None:
            json["active"] = active

        if continue_on_failure is not None:
            json["continue_on_failure"] = continue_on_failure

        if name is not None:
            json["name"] = name

        if oauth is not None:
            json["oauth"] = oauth

        if service_auth is not None:
            json["service_auth"] = service_auth

        if excludes is not None:
            json["excludes"] = excludes

        update_response = await self._request(f"{cohere.CONNECTOR_URL}/{id}", method="PATCH", json=json)
        return await self.get_connector(id=update_response["connector"]["id"])

    async def get_connector(self, id: str) -> Connector:
        """Returns a Connector given an id

        Args:
            id (str): The id of your connector

        Returns:
            Connector: Connector object.
        """
        if not id:
            raise CohereError(message="id must not be empty")
        response = await self._request(f"{cohere.CONNECTOR_URL}/{id}", method="GET")
        return Connector.from_dict(response["connector"])

    async def list_connectors(self, limit: int = None, offset: int = None) -> List[Connector]:
        """Returns a list of your Connectors

        Args:
            limit (int): (optional) The max number of connectors to return
            offset (int): (optional) The number of connectors to offset by

        Returns:
            List[Connector]: List of Connector objects.
        """
        param_dict = {}

        if limit is not None:
            param_dict["limit"] = limit

        if offset is not None:
            param_dict["offset"] = offset
        response = await self._request(f"{cohere.CONNECTOR_URL}", method="GET", params=param_dict)
        return [Connector.from_dict(r) for r in (response.get("connectors") or [])]

    async def delete_connector(self, id: str) -> None:
        """Deletes a Connector given an id

        Args:
            id (str): The id of your connector
        """
        if not id:
            raise CohereError(message="id must not be empty")
        await self._request(f"{cohere.CONNECTOR_URL}/{id}", method="DELETE")

    async def oauth_authorize_connector(self, id: str, after_token_redirect: str = None) -> str:
        """Returns a URL which when navigated to will start the OAuth 2.0 flow.

        Args:
            id (str): The id of your connector

        Returns:
            str: A URL that starts the OAuth 2.0 flow.
        """
        if not id:
            raise CohereError(message="id must not be empty")

        param_dict = {}

        if after_token_redirect is not None:
            param_dict["after_token_redirect"] = after_token_redirect

        response = await self._request(f"{cohere.CONNECTOR_URL}/{id}/oauth/authorize", method="GET", params=param_dict)
        return response["redirect_url"]


class AIOHTTPBackend:
    """HTTP backend which handles retries, concurrency limiting and logging"""

    SLEEP_AFTER_FAILURE = defaultdict(lambda: 0.25, {429: 5})

    def __init__(self, logger, max_concurrent_requests: int = 64, max_retries: int = 5, timeout: int = 300):
        self.logger = logger
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore: asyncio.Semaphore = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._requester = None

    def build_aio_requester(self) -> Callable:  # returns a function for retryable requests
        @backoff.on_exception(
            backoff.expo,
            (aiohttp.ClientError, aiohttp.ClientResponseError),
            max_tries=self.max_retries + 1,
            max_time=self.timeout,
        )
        async def make_request_fn(session, *args, **kwargs):
            async with self._semaphore:  # this limits total concurrency by the client
                response = await session.request(*args, **kwargs)
            if response.status in cohere.RETRY_STATUS_CODES:  # likely temporary, raise to retry
                self.logger.info(f"Received status {response.status}, retrying...")
                await asyncio.sleep(self.SLEEP_AFTER_FAILURE[response.status])
                response.raise_for_status()

            return response

        return make_request_fn

    async def request(
        self,
        url,
        json=None,
        files=None,
        method: str = "post",
        headers=None,
        session=None,
        stream=False,
        params=None,
        **kwargs,
    ) -> JSON:
        session = session or await self.session()
        self.logger.debug(f"Making request to {url} with content {json}")

        request_start = time.time()
        try:
            response = await self._requester(
                session, method, url, headers=headers, json=json, data=files, params=params, **kwargs
            )
        except aiohttp.ClientConnectionError as e:  # ensure the SDK user does not have to deal with knowing aiohttp
            self.logger.debug(f"Fatal connection error after {time.time()-request_start:.1f}s: {e}")
            raise CohereConnectionError(str(e)) from e
        except aiohttp.ClientResponseError as e:  # status 500 or something remains after retries
            self.logger.debug(f"Fatal ClientResponseError error after {time.time()-request_start:.1f}s: {e}")
            raise CohereConnectionError(str(e)) from e
        except asyncio.TimeoutError as e:
            self.logger.debug(f"Fatal timeout error after {time.time()-request_start:.1f}s: {e}")
            raise CohereConnectionError("The request timed out") from e
        except Exception as e:  # Anything caught here should be added above
            self.logger.debug(f"Unexpected fatal error after {time.time()-request_start:.1f}s: {e}")
            raise CohereError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

        self.logger.debug(f"Received response with status {response.status} after {time.time()-request_start:.1f}s")
        return response

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                json_serialize=np_json_dumps,
                timeout=aiohttp.ClientTimeout(self.timeout),
                connector=aiohttp.TCPConnector(limit=0),
            )
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self._requester = self.build_aio_requester()
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    def __del__(self):
        # https://stackoverflow.com/questions/54770360/how-can-i-wait-for-an-objects-del-to-finish-before-the-async-loop-closes
        if self._session:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass
