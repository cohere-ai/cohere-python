import asyncio
import os
from typing import Dict, List, Tuple, Union
from urllib.parse import urljoin

import numpy as np

from cohere.logging import logger
from cohere.responses import (
    BatchedGenerations,
    BatchedTokens,
    Embedding,
    Embeddings,
    Generations,
    Tokens,
)
import asyncio
import json
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import aiohttp
import backoff

from cohere.error import CohereAPIError, CohereConnectionError, CohereError
from cohere.logging import dummy_logger
from cohere.utils import np_json_dumps

JSON = Union[Dict, List]


def _ensure_list(items: Union[List, str]) -> Tuple[List[str], bool]:
    """Returns  a list of strings, and a bool indicating if it was a single item"""
    if isinstance(items, str):
        return [items], True
    else:
        return list(items), False


class AsyncClient:
    COHERE_VERSION = "2021-11-08"
    EMBED_BATCH_SIZE = 16

    CHECK_API_KEY_URL = "check-api-key"
    TOKENIZE_URL = "tokenize"
    DETOKENIZE_URL = "detokenize"
    GENERATE_URL = "generate"
    EMBED_URL = "embed"

    def __init__(
        self,
        api_key: str = None,
        max_concurrent_requests: int = 16,
        max_retries: int = 5,
        timeout: int = 120,
        return_exceptions: bool = False,
        api_url: str = None,
        client_name: str = None,
    ) -> None:
        """
        Args:
            api_key (str): your API key. Automatically taken from the CO_API_KEY environment variable if ommitted.
            max_concurrent_requests (int): maximum number of concurrent requests, shared between all calls to the client
            max_retries (int): maximum number of times to retry failing requests
            timeout (int): timeout for requests, including retries
            return_exceptions (bool): all calls with return CohereError object on exceptions, rather than raise. This is particularly useful for batched generations.
            api_url (str): override the api URL. Mainly for internal use.
            client_name (str): A string to identify your application for internal analytics purposes.
        """
        self.api_url = api_url or "https://api.cohere.ai"
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.request_source = "python-aio-sdk"
        if client_name is not None:
            self.request_source += ":" + client_name
        self._backend = AIOHTTPBackend(logger, max_concurrent_requests, max_retries, timeout)
        self.return_exceptions = return_exceptions

    async def _api_request(self, path, json=None, headers=None):
        headers = {
            "Authorization": f"BEARER {self.api_key}",
            "Request-Source": self.request_source,
            "Cohere-Version": self.COHERE_VERSION,
        }
        return await self._backend.request(urljoin(self.api_url, path), json, headers=headers)

    async def _batch_requests(self, path, jsons):  # NB: just raises first exception that occurs
        return await asyncio.gather(
            *(self._api_request(path, json) for json in jsons), return_exceptions=self.return_exceptions
        )

    async def close(self):
        return await self._backend.close()

    # API methods
    async def check_api_key(self) -> bool:
        return (await self._api_request(self.CHECK_API_KEY_URL))["valid"]

    async def embed(
        self, texts: Union[str, List[str]], model: str = None, truncate: str = "NONE"
    ) -> Union[Embedding, Embeddings]:
        texts, single_item = _ensure_list(texts)
        requests = [
            dict(texts=texts[i : i + self.EMBED_BATCH_SIZE], model=model, truncate=truncate)
            for i in range(0, len(texts), self.EMBED_BATCH_SIZE)
        ]
        responses = await self._batch_requests(self.EMBED_URL, requests)
        embeddings = Embeddings([e for res in responses for e in res["embeddings"]])  # concatenate results
        return embeddings[0] if single_item else embeddings

    async def tokenize(self, texts: Union[str, List[str]]) -> Union[Tokens, BatchedTokens]:
        texts, single_item = _ensure_list(texts)
        requests = [dict(text=text) for text in texts]
        responses = await self._batch_requests(self.TOKENIZE_URL, requests)
        tokens = [Tokens(tokens=r["tokens"], token_strings=r["token_strings"]) for r in responses]
        return tokens[0] if single_item else BatchedTokens(tokens)

    async def detokenize(self, items: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        if not items or not isinstance(items[0], (list, np.ndarray)):
            items = [items]
            single_item = True
        else:
            single_item = False
        requests = [dict(tokens=tokens) for tokens in items]
        responses = await self._batch_requests(self.DETOKENIZE_URL, requests)
        detokenizations = [r["text"] for r in responses]
        return detokenizations[0] if single_item else detokenizations

    async def generate(
        self,
        prompt: Union[str, List[str]] = None,
        model: str = None,
        preset: str = None,
        max_tokens: int = 10,  # api default is undocumented, but higher
        num_generations: int = 1,
        return_likelihoods: Union[str, bool] = None,
        prompt_vars: Dict[str, str] = None,
        **kwargs,
    ) -> Union[Generations, BatchedGenerations]:
        """Generate endpoint.
        See https://docs.cohere.ai/reference/generate for advanced arguments
        Args:
            * prompt (str): either a single prompt, or a list of them
            * return_likelihoods (str): One of GENERATION|ALL|NONE to specify how and if the token likelihoods are returned with the response.
                  Defaults to "NONE" for num_generations=1, or "GENERATION" otherwise, to ensure generations can be sorted by likelihood.
                  True/False are aliases for "GENERATION" and "NONE", respectively.
        Returns:
            * a Generations object, which functions as a list Generation objects
            * if multiple prompts are given, a list of the above

        """  # TODO: doc/bool truncate?
        if return_likelihoods is None:
            return_likelihoods = "GENERATION" if num_generations > 1 else "NONE"
        elif isinstance(return_likelihoods, bool):
            return_likelihoods = "GENERATION" if return_likelihoods else "NONE"

        prompts, single_item = _ensure_list(prompt)
        options = dict(
            model=model,
            preset=preset,
            max_tokens=max_tokens,
            num_generations=num_generations,
            return_likelihoods=return_likelihoods,
            prompt_vars=prompt_vars,
            **kwargs,
        )
        requests = [{"prompt": prompt, **options} for prompt in prompts]
        responses = await self._batch_requests(self.GENERATE_URL, requests)
        #TODO: no batching?
        results = [
            response if isinstance(response,Exception) else        Generations(return_likelihoods=return_likelihoods, response=response, client=self)
        
         for response in responses]
        return results[0] if single_item else BatchedGenerations(results)

    async def command(self, prompt, model="command-xlarge-20221108", max_tokens=250, *args, **kwargs):
        return await self.generate(prompt, model=model, max_tokens=max_tokens, *args, **kwargs)




class AIOHTTPBackend:
    """HTTP backend which handles retries, concurrency limiting and logging"""

    # TODO: should we retry error 500? Not normally, but I have seen them occurring intermittently.
    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
    SLEEP_AFTER_FAILURE = defaultdict(lambda: 0.25, {429: 1})

    def __init__(self, logger=None, max_concurrent_requests: int = 64, max_retries: int = 5, timeout: int = 120):
        self.logger = logger or dummy_logger
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

    async def request(self, url, json_payload=None, method: str = "post", headers=None, session=None, **kwargs) -> JSON:
        headers = {
            "Content-Type": "application/json",
            **(headers or {}),
        }
        session = session or await self.session()
        self.logger.debug(f"Making request to {url} with content {json_payload}")

        request_start = time.time()
        try:
            response = await self._requester(session, method, url, headers=headers, json=json_payload, **kwargs)
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
        except json.decoder.JSONDecodeError:  # CohereError will capture status
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
