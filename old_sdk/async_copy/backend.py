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
