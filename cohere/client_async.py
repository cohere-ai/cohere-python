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
from urllib.parse import urlparse

import requests

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
    DetectLanguageResponse,
    Detokenization,
    Embeddings,
    GenerateFeedbackResponse,
    GeneratePreferenceFeedbackResponse,
    Generations,
    LabelPrediction,
    Language,
    LogLikelihoods,
    PreferenceRating,
    Reranking,
    StreamingGenerations,
    SummarizeResponse,
    Tokens,
)
from cohere.responses.chat import AsyncChat, StreamingChat
from cohere.responses.classify import Example as ClassifyExample
from cohere.responses.cluster import AsyncClusterJobResult
from cohere.responses.custom_model import (
    CUSTOM_MODEL_PRODUCT_MAPPING,
    CUSTOM_MODEL_STATUS,
    CUSTOM_MODEL_TYPE,
    INTERNAL_CUSTOM_MODEL_TYPE,
    AsyncCustomModel,
    HyperParametersInput,
    ModelMetric,
)
from cohere.responses.dataset import AsyncDataset, BaseDataset, ParseInfo
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
            timeout=120,
            api_url: str = None,
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        if self.api_key != cohere.OCI_API_TYPE:
            self.api_url = api_url or os.getenv("CO_API_URL", cohere.COHERE_API_URL)
        else:
            self.api_url = api_url or os.getenv("CO_API_URL", cohere.OCI_COHERE_API_URL)
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk-" + cohere.SDK_VERSION
        self.max_retries = max_retries
        if client_name:
            self.request_source += ":" + client_name
        if self.api_key != cohere.OCI_API_TYPE:
            self.api_version = f"v{cohere.API_VERSION}"
        else:
            self.api_version = f"{cohere.OCI_API_VERSION}"
        self._check_api_key_on_enter = check_api_key
        if self.api_key != cohere.OCI_API_TYPE:
            self._backend = AIOHTTPBackend(logger, num_workers, max_retries, timeout)
        else:
            self._backend = OCIAIOHTTPBackend(logger, num_workers, max_retries, timeout)

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

    async def _oci_request(
            self, endpoint, json=None, files=None, method="POST", full_url=None, stream=False, params=None
    ) -> JSON:
        """refer: https://docs.oracle.com/en-us/iaas/Content/API/Concepts/signingrequests.htm#seven__Python"""
        from oci.config import from_file
        from oci.signer import Signer
        config = from_file("~/.oci/config")
        auth = Signer(
            tenancy=config['tenancy'],
            user=config['user'],
            fingerprint=config['fingerprint'],
            private_key_file_location=config['key_file'],
            pass_phrase=config['pass_phrase']
        )

        json.setdefault("compartmentId", config['tenancy'])

        if endpoint is None and full_url is not None:  # api key
            url = full_url
        else:
            url = posixpath.join(self.api_url, self.api_version, endpoint)

        response = await self._backend.request(url=url, json=json, method=method, auth=auth)
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
            logit_bias: Dict[int, float] = {},
            stream: bool = False,
    ) -> Union[Generations, StreamingGenerations]:
        if self.api_key != cohere.OCI_API_TYPE:
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
                "logit_bias": logit_bias,
                "stream": stream,
            }
        else:
            json_body = {
                "prompts": [prompt],
                "numGenerations": num_generations,
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topK": k,
                "topP": p,
                "frequencyPenalty": frequency_penalty,
                "presencePenalty": presence_penalty,
                "stopSequences": stop_sequences,
                "returnLikelihoods": return_likelihoods,
                "truncate": truncate,
                "isStream": stream,
                "isEcho": True,
                "servingMode": {"servingType": "ON_DEMAND", "modelId": model},
            }

        if self.api_key != cohere.OCI_API_TYPE:
            response = await self._request(cohere.GENERATE_URL, json=json_body, stream=stream)
        else:
            response = await self._oci_request(cohere.OCI_GENERATE_URL, json=json_body, stream=stream)

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
            preamble_override: Optional[str] = None,
            user_name: Optional[str] = None,
            temperature: Optional[float] = 0.8,
            max_tokens: Optional[int] = None,
            stream: Optional[bool] = False,
            p: Optional[float] = None,
            k: Optional[float] = None,
            logit_bias: Optional[Dict[int, float]] = None,
            search_queries_only: Optional[bool] = None,
            documents: Optional[List[Dict[str, Any]]] = None,
            citation_quality: Optional[str] = None,
            prompt_truncation: Optional[str] = None,
            connectors: Optional[List[Dict[str, Any]]] = None,
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
            "preamble_override": preamble_override,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "user_name": user_name,
            "p": p,
            "k": k,
            "logit_bias": logit_bias,
            "search_queries_only": search_queries_only,
            "documents": documents,
            "connectors": connectors,
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
            compress: Optional[bool] = False,
            compression_codebook: Optional[str] = "default",
            input_type: Optional[str] = None,
    ) -> Embeddings:
        """Returns an Embeddings object for the provided texts. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            text (List[str]): A list of strings to embed.
            model (str): (Optional) The model ID to use for embedding the text.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
            compress (bool): (Optional) Whether to compress the embeddings. When True, the compressed_embeddings will be returned as integers in the range [0, 255].
            compression_codebook (str): (Optional) The compression codebook to use for compressed embeddings. Defaults to "default".
            input_type (str): (Optional) One of "classification", "clustering", "search_document", "search_query". The type of input text provided to embed.
        """
        if self.api_key != cohere.OCI_API_TYPE:
            json_bodys = [
                dict(
                    texts=texts[i: i + cohere.COHERE_EMBED_BATCH_SIZE],
                    model=model,
                    truncate=truncate,
                    compress=compress,
                    compression_codebook=compression_codebook,
                    input_type=input_type,
                )
                for i in range(0, len(texts), cohere.COHERE_EMBED_BATCH_SIZE)
            ]
        else:
            json_bodys = [
                dict(
                    inputs=texts[i: i + cohere.COHERE_EMBED_BATCH_SIZE],
                    truncate=truncate,
                    isEcho=True,
                    servingMode={"servingType": "ON_DEMAND", "modelId": model},
                )
                for i in range(0, len(texts), cohere.COHERE_EMBED_BATCH_SIZE)
            ]

        if self.api_key != cohere.OCI_API_TYPE:
            responses = await asyncio.gather(*[self._request(cohere.EMBED_URL, json) for json in json_bodys])
        else:
            responses = await asyncio.gather(*[self._oci_request(cohere.OCI_EMBED_URL, json) for json in json_bodys])

        meta = responses[0]["meta"] if "meta" in responses[0].keys() else {"api_version": {"version": "1"}}

        if self.api_key != cohere.OCI_API_TYPE:
            return Embeddings(
                embeddings=[e for res in responses for e in res["embeddings"]],
                compressed_embeddings=[e for res in responses for e in
                                       res["compressed_embeddings"]] if compress else None,
                meta=meta,
            )
        else:
            return Embeddings(
                embeddings=[e for res in responses for e in res["embeddings"]],
                # compressed_embeddings=[e for res in responses for e in res["compressed_embeddings"]] if compress else None,
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
        if self.api_key != cohere.OCI_API_TYPE:
            json_body = {
                "model": model,
                "text": text,
                "length": length,
                "format": format,
                "temperature": temperature,
                "additional_command": additional_command,
                "extractiveness": extractiveness,
            }
        else:
            json_body = {
                "input": text,
                "length": length,
                "format": format,
                "temperature": temperature,
                "additionalCommand": additional_command,
                "extractiveness": extractiveness,
                "isEcho": True,
                "servingMode": {"servingType": "ON_DEMAND", "modelId": model},
            }
        # remove None values from the dict
        json_body = {k: v for k, v in json_body.items() if v is not None}
        if self.api_key != cohere.OCI_API_TYPE:
            response = await self._request(cohere.SUMMARIZE_URL, json=json_body)
        else:
            response = await self._oci_request(cohere.OCI_SUMMARIZE_URL, json=json_body)
            response["meta"] = {"api_version": {"version": "1"}}

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
        json_body = {
            "texts": texts,
        }
        response = await self._request(cohere.DETECT_LANG_URL, json=json_body)
        results = []
        for result in response["results"]:
            results.append(Language(result["language_code"], result["language_name"]))
        return DetectLanguageResponse(results, response["meta"])

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
        files = {"file": data}
        if eval_data:
            files["eval_file"] = eval_data
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
            input_dataset: Union[str, BaseDataset],
            name: Optional[str] = None,
            model: Optional[str] = None,
            truncate: Optional[str] = None,
            compress: Optional[bool] = None,
            compression_codebook: Optional[str] = None,
            text_field: Optional[str] = None,
    ) -> AsyncEmbedJob:
        """Create embed job.

        Args:
            input_dataset (Union[str, BaseDataset]): Dataset or dataset id with text to embed.
            name (Optional[str], optional): The name of the embed job. Defaults to None.
            model (Optional[str], optional): The model ID to use for embedding the text. Defaults to None.
            truncate (Optional[str], optional): How the API handles text longer than the maximum token length. Defaults to None.
            compress (Optional[bool], optional): Use embedding compression. Defaults to None.
            compression_codebook (Optional[str], optional): Embedding compression codebook. Defaults to None.
            text_field (Optional[str], optional): Name of the column containing text to embed. Defaults to None.

        Returns:
            AsyncEmbedJob: The created embed job
        """

        if isinstance(input_dataset, str):
            input_dataset_id = input_dataset
        elif isinstance(input_dataset, AsyncDataset):
            input_dataset_id = input_dataset.id
        else:
            raise CohereError(message="input_dataset must be either a string or Dataset")

        json_body = {
            "input_dataset_id": input_dataset_id,
            "name": name,
            "model": model,
            "truncate": truncate,
            "compress": compress,
            "compression_codebook": compression_codebook,
            "text_field": text_field,
            "output_format": "avro",
        }

        response = await self._request(cohere.EMBED_JOBS_URL, json=json_body)
        embed_job = await self.get_embed_job(response.get("job_id"))

        return embed_job

    async def list_embed_jobs(self) -> List[AsyncEmbedJob]:
        """List embed jobs.

        Returns:
            List[AsyncEmbedJob]: embed jobs.
        """

        response = await self._request(f"{cohere.EMBED_JOBS_URL}/list", method="GET")
        return [
            AsyncEmbedJob.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_embed_job)
            for r in response["bulk_embed_jobs"]
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
            dataset: CustomModelDataset,
            base_model: Optional[str] = None,
            hyperparameters: Optional[HyperParametersInput] = None,
    ) -> AsyncCustomModel:
        """Create a new custom model

        Args:
            name (str): name of your custom model, has to be unique across your organization
            model_type (GENERATIVE, CLASSIFY, RERANK): type of custom model
            dataset (InMemoryDataset, CsvDataset, JsonlDataset, TextDataset): A dataset for your training. Consists of a train and optional eval file.
            base_model (str): base model to use for your custom model.
                For generative and classify models, `base_model` has to be None (no option available for now)
                For rerank models, you can choose between `english` and `multilingual`. Defaults to `english` if not specified.
                    The English model is better for English, while the multilingual model should be picked if a non-negligible part of queries/documents
                    will be in other languages
            hyperparameters (HyperParametersInput): adjust hyperparameters for your custom model. Only for generative custom models.
        Returns:
            str: the id of the custom model that was created

        Examples:
            prompt completion custom model with csv file
                >>> from cohere.custom_model_dataset import CsvDataset
                >>> co = cohere.Client("YOUR_API_KEY")
                >>> dataset = CsvDataset(train_file="/path/to/your/file.csv", delimiter=",")
                >>> finetune = co.create_custom_model("prompt-completion-ft", dataset=dataset, model_type="GENERATIVE")

            prompt completion custom model with in-memory dataset
                >>> from cohere.custom_model_dataset import InMemoryDataset
                >>> co = cohere.Client("YOUR_API_KEY")
                >>> dataset = InMemoryDataset(training_data=[
                >>>     ("this is the prompt", "and this is the completion"),
                >>>     ("another prompt", "and another completion")
                >>> ])
                >>> finetune = co.create_custom_model("prompt-completion-ft", dataset=dataset, model_type="GENERATIVE")

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
                "trainSteps": hyperparameters.get("train_steps"),
                "learningRate": hyperparameters.get("learning_rate"),
            }

        remote_path = await self._upload_dataset(
            dataset.get_train_data(), name, dataset.train_file_name(), internal_custom_model_type
        )
        json["settings"]["trainFiles"].append({"path": remote_path, **dataset.file_config()})
        if dataset.has_eval_file():
            remote_path = await self._upload_dataset(
                dataset.get_eval_data(), name, dataset.eval_file_name(), internal_custom_model_type
            )
            json["settings"]["evalFiles"].append({"path": remote_path, **dataset.file_config()})

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
            statuses (CUSTOM_MODEL_STATUS, optional): search for fintunes which are in one of these states
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

        json = {
            "query": {
                "statuses": statuses,
                "before": before.isoformat(timespec="seconds") if before else None,
                "after": after.isoformat(timespec="seconds") if after else None,
                "orderBy": order_by,
            }
        }

        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/ListFinetunes", method="POST", json=json)
        return [AsyncCustomModel.from_dict(r, self.wait_for_custom_model) for r in response["finetunes"]]


class AIOHTTPBackend:
    """HTTP backend which handles retries, concurrency limiting and logging"""

    SLEEP_AFTER_FAILURE = defaultdict(lambda: 0.25, {429: 5})

    def __init__(self, logger, max_concurrent_requests: int = 64, max_retries: int = 5, timeout: int = 120):
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
            self.logger.debug(f"Fatal connection error after {time.time() - request_start:.1f}s: {e}")
            raise CohereConnectionError(str(e)) from e
        except aiohttp.ClientResponseError as e:  # status 500 or something remains after retries
            self.logger.debug(f"Fatal ClientResponseError error after {time.time() - request_start:.1f}s: {e}")
            raise CohereConnectionError(str(e)) from e
        except asyncio.TimeoutError as e:
            self.logger.debug(f"Fatal timeout error after {time.time() - request_start:.1f}s: {e}")
            raise CohereConnectionError("The request timed out") from e
        except Exception as e:  # Anything caught here should be added above
            self.logger.debug(f"Unexpected fatal error after {time.time() - request_start:.1f}s: {e}")
            raise CohereError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

        self.logger.debug(f"Received response with status {response.status} after {time.time() - request_start:.1f}s")
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


class OCIAIOHTTPBackend:
    """HTTP backend which handles retries, concurrency limiting and logging"""

    SLEEP_AFTER_FAILURE = defaultdict(lambda: 0.25, {429: 5})

    def __init__(self, logger, max_concurrent_requests: int = 64, max_retries: int = 5, timeout: int = 120):
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
            method: str = "post",
            session=None,
            auth=None,
    ) -> JSON:
        session = session or await self.session()
        self.logger.debug(f"Making request to {url} with content {json}")

        request_start = time.time()
        try:
            auth_req = requests.Request("POST", url, json=json, auth=auth)
            auth_req.body = jsonlib.dumps(json)
            auth_req.path_url = urlparse(url).path
            auth_req = auth(auth_req)

            response = await self._requester(
                session=session, method=method, url=url, headers=auth_req.headers, json=json
            )
        except aiohttp.ClientConnectionError as e:  # ensure the SDK user does not have to deal with knowing aiohttp
            self.logger.debug(f"Fatal connection error after {time.time() - request_start:.1f}s: {e}")
            raise CohereConnectionError(str(e)) from e
        except aiohttp.ClientResponseError as e:  # status 500 or something remains after retries
            self.logger.debug(f"Fatal ClientResponseError error after {time.time() - request_start:.1f}s: {e}")
            raise CohereConnectionError(str(e)) from e
        except asyncio.TimeoutError as e:
            self.logger.debug(f"Fatal timeout error after {time.time() - request_start:.1f}s: {e}")
            raise CohereConnectionError("The request timed out") from e
        except Exception as e:  # Anything caught here should be added above
            self.logger.debug(f"Unexpected fatal error after {time.time() - request_start:.1f}s: {e}")
            raise CohereError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

        self.logger.debug(f"Received response with status {response.status} after {time.time() - request_start:.1f}s")
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
