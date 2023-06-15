import asyncio
import json as jsonlib
import os
import posixpath
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

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
    AsyncCreateClusterJobResponse,
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
    PreferenceRating,
    Reranking,
    StreamingGenerations,
    SummarizeResponse,
    Tokens,
)
from cohere.responses.bulk_embed import AsyncCreateBulkEmbedJobResponse, BulkEmbedJob
from cohere.responses.chat import AsyncChat, StreamingChat
from cohere.responses.classify import Example as ClassifyExample
from cohere.responses.custom_model import (
    CUSTOM_MODEL_PRODUCT_MAPPING,
    CUSTOM_MODEL_STATUS,
    CUSTOM_MODEL_TYPE,
    INTERNAL_CUSTOM_MODEL_TYPE,
    CustomModel,
)
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
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = os.getenv("CO_API_URL", cohere.COHERE_API_URL)
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

    async def _request(self, endpoint, json=None, method="POST", full_url=None, stream=False) -> JSON:
        headers = {
            "Authorization": f"BEARER {self.api_key}",
            "Request-Source": self.request_source,
        }
        if endpoint is None and full_url is not None:  # api key
            url = full_url
        else:
            url = posixpath.join(self.api_url, self.api_version, endpoint)

        response = await self._backend.request(url, json, method, headers, stream=stream)
        if stream:
            return response

        try:
            json_response = await response.json()
        #   `CohereAPIError.from_response()` will capture the http status code
        except jsonlib.decoder.JSONDecodeError:
            raise CohereAPIError.from_response(response, message=f"Failed to decode json body: {await response.text()}")
        except aiohttp.ClientPayloadError as e:
            raise CohereAPIError.from_response(
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
        response = await self._request(cohere.GENERATE_URL, json=json_body, stream=stream)
        if stream:
            return StreamingGenerations(response)
        else:
            return Generations.from_dict(response=response, return_likelihoods=return_likelihoods)

    async def chat(
        self,
        query: str,
        conversation_id: Optional[str] = "",
        model: Optional[str] = None,
        return_chatlog: Optional[bool] = False,
        return_prompt: Optional[bool] = False,
        return_preamble: Optional[bool] = False,
        chatlog_override: List[Dict[str, str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        preamble_override: Optional[str] = None,
        user_name: Optional[str] = None,
        temperature: Optional[float] = 0.8,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        p: Optional[float] = None,
        k: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Union[AsyncChat, StreamingChat]:
        if chatlog_override is not None:
            logger.warning(
                "The 'chatlog_override' parameter is deprecated and will be removed in a future version of this function. "
                + "Use 'chat_history' to keep track of the conversation instead."
            )

        if chat_history is not None:
            self._validate_chat_history(chat_history)

        json_body = {
            "query": query,
            "conversation_id": conversation_id,
            "model": model,
            "return_chatlog": return_chatlog,
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
        }

        response = await self._request(cohere.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return AsyncChat.from_dict(response, query=query, client=self)

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        truncate: Optional[str] = None,
        compress: Optional[bool] = False,
        compression_codebook: Optional[str] = "default",
    ) -> Embeddings:
        """Returns an Embeddings object for the provided texts. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            text (List[str]): A list of strings to embed.
            model (str): (Optional) The model ID to use for embedding the text.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
            compress (bool): (Optional) Whether to compress the embeddings. When True, the compressed_embeddings will be returned as integers in the range [0, 255].
            compression_codebook (str): (Optional) The compression codebook to use for compressed embeddings. Defaults to "default".
        """
        json_bodys = [
            dict(
                texts=texts[i : i + cohere.COHERE_EMBED_BATCH_SIZE],
                model=model,
                truncate=truncate,
                compress=compress,
                compression_codebook=compression_codebook,
            )
            for i in range(0, len(texts), cohere.COHERE_EMBED_BATCH_SIZE)
        ]
        responses = await asyncio.gather(*[self._request(cohere.EMBED_URL, json) for json in json_bodys])
        meta = responses[0]["meta"] if responses else None

        return Embeddings(
            embeddings=[e for res in responses for e in res["embeddings"]],
            compressed_embeddings=[e for res in responses for e in res["compressed_embeddings"]] if compress else None,
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
                Classification(res["input"], res["prediction"], res["confidence"], labelObj, id=res["id"])
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

    async def create_cluster_job(
        self,
        embeddings_url: str,
        min_cluster_size: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        is_deterministic: Optional[bool] = None,
    ) -> AsyncCreateClusterJobResponse:
        """Create clustering job.

        Args:
            embeddings_url (str): File with embeddings to cluster.
            min_cluster_size (Optional[int], optional): Minimum number of elements in a cluster. Defaults to 10.
            n_neighbors (Optional[int], optional): Number of nearest neighbors used by UMAP to establish the
                local structure of the data. Defaults to 15. For more information, please refer to
                https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
            is_deterministic (Optional[bool], optional): Determines whether the output of the cluster job is
                deterministic. Defaults to True.

        Returns:
            CreateClusterJobResponse: Created clustering job handler
        """

        json_body = {
            "embeddings_url": embeddings_url,
            "min_cluster_size": min_cluster_size,
            "n_neighbors": n_neighbors,
            "is_deterministic": is_deterministic,
        }
        response = await self._request(cohere.CLUSTER_JOBS_URL, json=json_body)
        return AsyncCreateClusterJobResponse.from_dict(
            response,
            wait_fn=self.wait_for_cluster_job,
        )

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
        return ClusterJobResult.from_dict(response)

    async def list_cluster_jobs(self) -> List[ClusterJobResult]:
        """List clustering jobs.

        Returns:
            List[ClusterJobResult]: Clustering jobs created.
        """

        response = await self._request(cohere.CLUSTER_JOBS_URL, method="GET")
        return [ClusterJobResult.from_dict({"meta": response.get("meta"), **r}) for r in response["jobs"]]

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

    async def create_bulk_embed_job(
        self,
        input_file_url: str,
        model: Optional[str] = None,
        truncate: Optional[str] = None,
        compress: Optional[bool] = None,
        compression_codebook: Optional[str] = None,
        text_field: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> AsyncCreateBulkEmbedJobResponse:
        """Create bulk embed job.

        Args:
            input_file_url (str): File with texts to embed.
            model (Optional[str], optional): The model ID to use for embedding the text. Defaults to None.
            truncate (Optional[str], optional): How the API handles text longer than the maximum token length. Defaults to None.
            compress (Optional[bool], optional): Use embedding compression. Defaults to None.
            compression_codebook (Optional[str], optional): Embedding compression codebook. Defaults to None.
            text_field (Optional[str], optional): Name of the column containing text to embed. Defaults to None.
            output_format (Optional[str], optional): Output format and file extension. Defaults to None.

        Returns:
            AsyncCreateBulkEmbedJobResponse: Created bulk embed job handler
        """

        json_body = {
            "input_file_url": input_file_url,
            "model": model,
            "truncate": truncate,
            "compress": compress,
            "compression_codebook": compression_codebook,
            "text_field": text_field,
            "output_format": output_format,
        }

        response = await self._request(cohere.BULK_EMBED_JOBS_URL, json=json_body)

        return AsyncCreateBulkEmbedJobResponse.from_dict(
            response,
            wait_fn=self.wait_for_bulk_embed_job,
        )

    async def list_bulk_embed_jobs(self) -> List[BulkEmbedJob]:
        """List bulk embed jobs.

        Returns:
            List[BulkEmbedJob]: Bulk embed jobs.
        """

        response = await self._request(f"{cohere.BULK_EMBED_JOBS_URL}/list", method="GET")
        return [BulkEmbedJob.from_dict({"meta": response.get("meta"), **r}) for r in response["bulk_embed_jobs"]]

    async def get_bulk_embed_job(self, job_id: str) -> BulkEmbedJob:
        """Get bulk embed job.

        Args:
            job_id (str): Bulk embed job id.

        Raises:
            ValueError: "job_id" is empty

        Returns:
            BulkEmbedJob: Bulk embed job.
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        response = await self._request(f"{cohere.BULK_EMBED_JOBS_URL}/{job_id}", method="GET")
        return BulkEmbedJob.from_dict(response)

    async def cancel_bulk_embed_job(self, job_id: str) -> None:
        """Cancel bulk embed job.

        Args:
            job_id (str): Bulk embed job id.

        Raises:
            ValueError: "job_id" is empty
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        await self._request(f"{cohere.BULK_EMBED_JOBS_URL}/{job_id}/cancel", method="POST", json={})

    async def wait_for_bulk_embed_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> BulkEmbedJob:
        """Wait for bulk embed job completion.

        Args:
            job_id (str): Bulk embed job id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            BulkEmbedJob: Bulk embed job.
        """

        return await async_wait_for_job(
            get_job=partial(self.get_bulk_embed_job, job_id),
            timeout=timeout,
            interval=interval,
        )

    async def create_custom_model(
        self, name: str, model_type: CUSTOM_MODEL_TYPE, dataset: CustomModelDataset
    ) -> CustomModel:
        """Create a new custom model

        Args:
            name (str): name of your custom model, has to be unique across your organization
            model_type (GENERATIVE, EMBED, CLASSIFY): type of custom model
            dataset (InMemoryDataset, CsvDataset, JsonlDataset, TextDataset): A dataset for your training. Consists of a train and optional eval file.
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
                "baseModel": "medium",
                "finetuneType": internal_custom_model_type,
            },
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
        return CustomModel.from_dict(response["finetune"])

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

    async def get_custom_model(self, custom_model_id: str) -> CustomModel:
        """Get a custom model by id.

        Args:
            custom_model_id (str): custom model id
        Returns:
            CustomModel: the custom model
        """
        json = {"finetuneID": custom_model_id}
        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetune", method="POST", json=json)
        return CustomModel.from_dict(response["finetune"])

    async def get_custom_model_by_name(self, name: str) -> CustomModel:
        """Get a custom model by name.

        Args:
            name (str): custom model name
        Returns:
            CustomModel: the custom model
        """
        json = {"name": name}
        response = await self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneByName", method="POST", json=json)
        return CustomModel.from_dict(response["finetune"])

    async def list_custom_models(
        self,
        statuses: Optional[List[CUSTOM_MODEL_STATUS]] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        order_by: Optional[Literal["asc", "desc"]] = None,
    ) -> List[CustomModel]:
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
        return [CustomModel.from_dict(r) for r in response["finetunes"]]


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
        self, url, json=None, method: str = "post", headers=None, session=None, stream=False, **kwargs
    ) -> JSON:
        session = session or await self.session()
        self.logger.debug(f"Making request to {url} with content {json}")

        request_start = time.time()
        try:
            response = await self._requester(session, method, url, headers=headers, json=json, **kwargs)
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
