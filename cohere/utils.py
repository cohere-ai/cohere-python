import abc
import asyncio
import json
import time
import typing
from typing import Awaitable, Callable, Optional, TypeVar

from cohere.error import CohereError

try:  # numpy is optional, but support json encoding if the user has it
    import numpy as np

    class CohereJsonEncoder(json.JSONEncoder):
        """Handles numpy datatypes and such in json encoding"""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            else:
                return super().default(obj)

except:

    class CohereJsonEncoder(json.JSONEncoder):
        """numpy is missing, so we can't handle these (and don't expect them)"""

        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            else:
                return super().default(obj)


def np_json_dumps(data, **kwargs):
    return json.dumps(data, cls=CohereJsonEncoder, **kwargs)


def is_api_key_valid(key: typing.Optional[str]) -> bool:
    """is_api_key_valid returns True when the key is valid and raises a CohereError when it is invalid."""
    if not key:
        raise CohereError(
            "No API key provided. Provide the API key in the client initialization or the CO_API_KEY environment variable."  # noqa: E501
        )

    return True


class JobWithStatus(abc.ABC):
    @abc.abstractmethod
    def has_terminal_status(self) -> bool:
        ...


T = TypeVar("T", bound=JobWithStatus)


def wait_for_job(
    get_job: Callable[[], T],
    timeout: Optional[float] = None,
    interval: float = 10,
) -> T:
    start_time = time.time()
    job = get_job()

    while not job.has_terminal_status():
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"wait_for_job timed out after {timeout} seconds")

        time.sleep(interval)
        job = get_job()

    return job


async def async_wait_for_job(
    get_job: Callable[[], Awaitable[T]],
    timeout: Optional[float] = None,
    interval: float = 10,
) -> T:
    start_time = time.time()
    job = await get_job()

    while not job.has_terminal_status():
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"wait_for_job timed out after {timeout} seconds")

        await asyncio.sleep(interval)
        job = await get_job()

    return job
