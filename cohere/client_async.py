import asyncio
import json as jsonlib
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp
import backoff

import cohere
from cohere.client import Client
from cohere.error import CohereAPIError, CohereConnectionError, CohereError
from cohere.logging import logger
from cohere.responses.base import CohereObject
from cohere.responses.chat import AsyncChat
from cohere.responses import (
    Classification,
    Classifications,
    DetectLanguageResponse,
    Detokenization,
    Embeddings,
    Feedback,
    Generations,
    LabelPrediction,
    Language,
    Reranking,
    SummarizeResponse,
    Tokens,
)
from cohere.responses.cluster import AsyncCreateClusterJobResponse, ClusterJobResult
from cohere.responses.classify import Example as ClassifyExample
from cohere.utils import np_json_dumps

JSON = Union[Dict, List]


class AsyncClient(Client):
    """AsyncClient
    
    This client provides an asyncio/aiohttp interface.
    Using this client is recommended when you are making highly parallel request,
    or when calling the Cohere API from a server such as FastAPI.

    The methods here are typically identical to those in the main `Client`, with an extra argument 
    `return_exceptions` for the batch* methods, which is passed to asyncio.gather"""

    def __init__(
        self,
        api_key: str = None,
        version: Optional[str] = None,
        num_workers: int = 16,
        request_dict: dict = {},
        check_api_key: bool = False,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout=120,
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk"
        self.max_retries = max_retries
        if client_name:
            self.request_source += ":" + client_name
        self.cohere_version = version or cohere.COHERE_VERSION
        self._need_to_check_api_key = check_api_key  # TODO: check in __enter__
        self._backend = AIOHTTPBackend(logger, num_workers, max_retries, timeout)

    async def __request(self, path, json=None, method="POST") -> JSON:
        headers = {
            "Authorization": f"BEARER {self.api_key}",
            "Request-Source": self.request_source,
            "Cohere-Version": self.cohere_version,
        }
        return await self._backend.request(urljoin(self.api_url, path), json, method,headers)

    async def close(self):
        return await self._backend.close()

    # API methods
    async def check_api_key(self) -> Dict[str, bool]:
        return await self.__request(cohere.CHECK_API_KEY_URL)

    async def batch_generate(self, prompts: List[str], return_exceptions=False, **kwargs) -> List[Generations]:
        """return_exceptions is passed to asyncio.gather"""
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
    ) -> Generations:
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
        }
        response = await self.__request(cohere.GENERATE_URL, json_body)
        return Generations(return_likelihoods=return_likelihoods, response=response)

    async def chat(
        self,
        query: str,
        session_id: str = "",
        persona: str = "cohere",
        model: Optional[str] = None,
        return_chatlog: bool = False,
        chatlog_override: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncChat:

        if chatlog_override is not None:
            self._validate_chatlog_override(chatlog_override)

        json_body = {
            "query": query,
            "session_id": session_id,
            "persona": persona,
            "model": model,
            "return_chatlog": return_chatlog,
            "chatlog_override": chatlog_override,
        }
        response = await self.__request(cohere.CHAT_URL, json=json_body)
        return AsyncChat(query=query, persona=persona, response=response, return_chatlog=return_chatlog, client=self)

    async def embed(self, texts: List[str], model: Optional[str] = None, truncate: Optional[str] = None) -> Embeddings:
        json_bodys = [
            dict(texts=texts[i : i + cohere.COHERE_EMBED_BATCH_SIZE], model=model, truncate=truncate)
            for i in range(0, len(texts), cohere.COHERE_EMBED_BATCH_SIZE)
        ]
        responses = await asyncio.gather(*[self.__request(cohere.EMBED_URL, json) for json in json_bodys])
        embeddings = Embeddings([e for res in responses for e in res["embeddings"]])  # concatenate results
        return embeddings

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
        response = await self.__request(cohere.CLASSIFY_URL, json=json_body)
        classifications = []
        for res in response["classifications"]:
            labelObj = {}
            for label, prediction in res["labels"].items():
                labelObj[label] = LabelPrediction(prediction["confidence"])
            classifications.append(
                Classification(res["input"], res["prediction"], res["confidence"], labelObj, id=res["id"])
            )

        return Classifications(classifications)

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
        response = await self.__request(cohere.SUMMARIZE_URL, json=json_body)
        return SummarizeResponse(id=response["id"], summary=response["summary"])

    async def batch_tokenize(self, texts: List[str], return_exceptions=False) -> List[Tokens]:
        return await asyncio.gather(*[self.tokenize(t) for t in texts], return_exceptions=return_exceptions)

    async def tokenize(self, text: str) -> Tokens:
        json_body = {"text": text}
        res = await self.__request(cohere.TOKENIZE_URL, json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"])

    async def batch_detokenize(self, list_of_tokens: List[List[int]], return_exceptions=False) -> List[Detokenization]:
        return await asyncio.gather(*[self.detokenize(t) for t in list_of_tokens], return_exceptions=return_exceptions)

    async def detokenize(self, tokens: List[int]) -> Detokenization:
        json_body = {"tokens": tokens}
        res = await self.__request(cohere.DETOKENIZE_URL, json_body)
        return Detokenization(text=res["text"])

    async def detect_language(self, texts: List[str]) -> DetectLanguageResponse:
        json_body = {
            "texts": texts,
        }
        response = await self.__request(cohere.DETECT_LANG_URL, json=json_body)
        results = []
        for result in response["results"]:
            results.append(Language(result["language_code"], result["language_name"]))
        return DetectLanguageResponse(results)

    async def feedback(self, id: str, good_response: bool, desired_response: str = "", feedback: str = "") -> Feedback:
        json_body = {
            "id": id,
            "good_response": good_response,
            "desired_response": desired_response,
            "feedback": feedback,
        }
        await self.__request(cohere.FEEDBACK_URL, json_body)
        return Feedback(id=id, good_response=good_response, desired_response=desired_response, feedback=feedback)

    async def rerank(
        self, query: str, documents: Union[List[str], List[Dict[str, Any]]], top_n: Optional[int] = None
    ) -> Reranking:
        """Returns an ordered list of documents ordered by their relevance to the provided query

        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            top_n (int): (optional) The number of results to return, defaults to returning all results
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
            "top_n": top_n,
            "return_documents": False,
        }
        reranking = Reranking(await self.__request(cohere.RERANK_URL, json=json_body))
        for rank in reranking.results:
            rank.document = parsed_docs[rank.index]
        return reranking

    async def create_cluster_job(
        self,
        embeddings_url: str,
        threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
    ) -> AsyncCreateClusterJobResponse:
        json_body = {
            "embeddings_url": embeddings_url,
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
        }
        response = await self.__request(cohere.CLUSTER_JOBS_URL, json=json_body)
        return AsyncCreateClusterJobResponse(
            job_id=response['job_id'],
            wait_fn=self.wait_for_cluster_job,
        )

    async def get_cluster_job(
        self,
        job_id: str,
    ) -> ClusterJobResult:
        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        response = await self.__request(os.path.join(cohere.CLUSTER_JOBS_URL, job_id), method='GET')
        return ClusterJobResult(
            status=response['status'],
            output_clusters_url=response['output_clusters_url'],
            output_outliers_url=response['output_outliers_url'],
        )

    async def wait_for_cluster_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> ClusterJobResult:
        start_time = time.time()
        job = await self.get_cluster_job(job_id)

        while job.status == 'processing':
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f'wait_for_cluster_job timed out after {timeout} seconds')

            await asyncio.sleep(interval)
            job = await self.get_cluster_job(job_id)

        return job

    async def wait_for_cluster_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> ClusterJobResult:
        start_time = time.time()
        job = await self.get_cluster_job(job_id)

        while job.status == 'processing':
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f'wait_for_cluster_job timed out after {timeout} seconds')

            asyncio.sleep(interval)
            job = await self.get_cluster_job(job_id)

        return job



class AIOHTTPBackend:
    """HTTP backend which handles retries, concurrency limiting and logging"""

    # TODO: should we retry error 500? Not normally, but I have seen them occurring intermittently.
    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
    SLEEP_AFTER_FAILURE = defaultdict(lambda: 0.25, {429: 1})

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
            if response.status in self.RETRY_STATUS_CODES:  # likely temporary, raise to retry
                self.logger.info(f"Received status {response.status}, retrying...")
                await asyncio.sleep(self.SLEEP_AFTER_FAILURE[response.status])
                response.raise_for_status()

            return response

        return make_request_fn

    async def request(self, url, json=None, method: str = "post", headers=None, session=None, **kwargs) -> JSON:
        headers = {
            "Content-Type": "application/json",
            **(headers or {}),
        }
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

        if "X-API-Warning" in response.headers:
            self.logger.warning(response.headers["X-API-Warning"])

        try:
            json_response = await response.json()
        except jsonlib.decoder.JSONDecodeError:  # CohereError will capture status
            raise CohereAPIError.from_response(response, message=f"Failed to decode json body: {await response.text()}")
        self.logger.debug(
            f"Received response with status {response.status} after {time.time()-request_start:.1f}s : {json_response}"
        )
        if "message" in json_response:  # has errors
            raise CohereAPIError.from_response(response, message=json_response["message"])
        return json_response

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
