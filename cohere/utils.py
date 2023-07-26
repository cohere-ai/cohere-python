import abc
import asyncio
import json
import time
from concurrent import futures
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional, TypeVar

from cohere.error import CohereError
from cohere.logging import logger

datetime_fmt = "%Y-%m-%dT%H:%M:%SZ"
datetime_fmt_with_milli = "%Y-%m-%dT%H:%M:%S.%fZ"

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


def is_api_key_valid(key: Optional[str]) -> bool:
    """is_api_key_valid returns True when the key is valid and raises a CohereError when it is invalid."""
    if not key:
        raise CohereError(
            "No API key provided. Provide the API key in the client initialization or the CO_API_KEY environment variable."  # noqa: E501
        )

    return True


def parse_datetime(datetime_str) -> datetime:
    try:
        return datetime.strptime(datetime_str, datetime_fmt_with_milli)
    except:
        return datetime.strptime(datetime_str, datetime_fmt)


class JobWithStatus(abc.ABC):
    @abc.abstractmethod
    def has_terminal_status(self) -> bool:
        ...

    @abc.abstractmethod
    def wait(self, timeout: Optional[float] = None, interval: float = 10) -> "JobWithStatus":
        ...

    def _update_self(self, updated_job):
        for k, v in updated_job.__dict__.items():
            setattr(self, k, v)


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
        logger.warning("...")
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


def threadpool_map(f, call_args: List[Dict], num_workers, return_exceptions: bool = False) -> List:
    """Helper function similar to futures.ThreadPoolExecutor.map, but allows returning exceptions"""
    results = []
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_list = [executor.submit(f, **args) for args in call_args]
        for future in futures_list:
            try:
                results.append(future.result())
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise
    return results
