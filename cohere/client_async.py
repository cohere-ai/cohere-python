import asyncio
import json as jsonlib
import os
import posixpath
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import backoff

import cohere
from cohere.client import Client
from cohere.error import CohereAPIError, CohereConnectionError, CohereError
from cohere.logging import logger
from cohere.responses import (
    AsyncCreateClusterJobResponse,
    Classification,
    Classifications,
    ClusterJobResult,
    DetectLanguageResponse,
    Detokenization,
    Embeddings,
    GenerateFeedbackResponse,
    Generations,
    LabelPrediction,
    Language,
    Reranking,
    StreamingGenerations,
    SummarizeResponse,
    Tokens,
)
from cohere.responses.chat import AsyncChat, StreamingChat
from cohere.responses.classify import Example as ClassifyExample
from cohere.utils import is_api_key_valid, np_json_dumps

JSON = Union[Dict, List]


class AsyncClient(Client):
    """AsyncClient

    This client provides an asyncio/aiohttp interface.
    Using this client is recommended when you are making highly parallel request,
    or when calling the Cohere API from a server such as FastAPI.

    The methods here are typically identical to those in the main `Client`, with an extra argument
    `return_exceptions` for the batch* methods, which is passed to asyncio.gather."""

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
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk"
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
        except jsonlib.decoder.JSONDecodeError:  # CohereAPIError will capture status
            raise CohereAPIError.from_response(response, message=f"Failed to decode json body: {await response.text()}")

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
        session_id: str = "",
        conversation_id: str = "",
        model: Optional[str] = None,
        return_chatlog: bool = False,
        return_prompt: bool = False,
        return_preamble: bool = False,
        chatlog_override: List[Dict[str, str]] = None,
        persona_name: str = None,
        persona_prompt: str = None,
        preamble_override: str = None,
        user_name: str = None,
        temperature: float = 0.8,
        max_tokens: int = 200,
        stream: bool = False,
    ) -> Union[AsyncChat, StreamingChat]:
        if chatlog_override is not None:
            self._validate_chatlog_override(chatlog_override)

        if session_id != "":
            conversation_id = session_id
            logger.warning(
                "The 'session_id' parameter is deprecated and will be removed in a future version of this function. Use 'conversation_id' instead.",
            )
        if persona_prompt is not None:
            preamble_override = persona_prompt
            logger.warning(
                "The 'persona_prompt' parameter is deprecated and will be removed in a future version of this function. Use 'preamble_override' instead.",
            )
        if persona_name is not None:
            logger.warning(
                "The 'persona_name' parameter is deprecated and will be removed in a future version of this function.",
            )
        if user_name is not None:
            logger.warning(
                "The 'user_name' parameter is deprecated and will be removed in a future version of this function.",
            )

        json_body = {
            "query": query,
            "conversation_id": conversation_id,
            "model": model,
            "return_chatlog": return_chatlog,
            "return_prompt": return_prompt,
            "return_preamble": return_preamble,
            "chatlog_override": chatlog_override,
            "preamble_override": preamble_override,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        response = await self._request(cohere.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return AsyncChat.from_dict(response, query=query, client=self)

    async def embed(self, texts: List[str], model: Optional[str] = None, truncate: Optional[str] = None) -> Embeddings:
        json_bodys = [
            dict(texts=texts[i : i + cohere.COHERE_EMBED_BATCH_SIZE], model=model, truncate=truncate)
            for i in range(0, len(texts), cohere.COHERE_EMBED_BATCH_SIZE)
        ]
        responses = await asyncio.gather(*[self._request(cohere.EMBED_URL, json) for json in json_bodys])
        meta = responses[0]["meta"] if responses else None

        embeddings = Embeddings([e for res in responses for e in res["embeddings"]], meta)  # concatenate results
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

    async def batch_tokenize(self, texts: List[str], return_exceptions=False) -> List[Tokens]:
        return await asyncio.gather(*[self.tokenize(t) for t in texts], return_exceptions=return_exceptions)

    async def tokenize(self, text: str) -> Tokens:
        json_body = {"text": text}
        res = await self._request(cohere.TOKENIZE_URL, json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"], meta=res["meta"])

    async def batch_detokenize(self, list_of_tokens: List[List[int]], return_exceptions=False) -> List[Detokenization]:
        return await asyncio.gather(*[self.detokenize(t) for t in list_of_tokens], return_exceptions=return_exceptions)

    async def detokenize(self, tokens: List[int]) -> Detokenization:
        json_body = {"tokens": tokens}
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

    async def rerank(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        model: str = None,
        top_n: Optional[int] = None,
    ) -> Reranking:
        """Returns an ordered list of documents ordered by their relevance to the provided query

        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            model (str): (Optional) The model to use for re-ranking
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
            "model": model,
            "top_n": top_n,
            "return_documents": False,
        }
        reranking = Reranking(await self._request(cohere.RERANK_URL, json=json_body))
        for rank in reranking.results:
            rank.document = parsed_docs[rank.index]
        return reranking

    async def create_cluster_job(
        self,
        embeddings_url: str,
        threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
    ) -> AsyncCreateClusterJobResponse:
        """Create clustering job.

        Args:
            embeddings_url (str): File with embeddings to cluster.
            threshold (Optional[float], optional): Similarity threshold above which two texts are deemed to belong in
                the same cluster. Defaults to None.
            min_cluster_size (Optional[int], optional): Minimum number of elements in a cluster. Defaults to None.

        Returns:
            CreateClusterJobResponse: Created clustering job handler
        """

        json_body = {
            "embeddings_url": embeddings_url,
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
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

        start_time = time.time()
        job = await self.get_cluster_job(job_id)

        while not job.is_final_state:
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"wait_for_cluster_job timed out after {timeout} seconds")

            await asyncio.sleep(interval)
            job = await self.get_cluster_job(job_id)

        return job


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
