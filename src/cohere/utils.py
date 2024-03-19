import asyncio
import time
import typing
from typing import Optional

from .types import EmbedJob, CreateEmbedJobResponse
from .datasets import DatasetsCreateResponse, DatasetsGetResponse


def get_terminal_states():
    return get_success_states() | get_failed_states()


def get_success_states():
    return {"complete", "validated"}


def get_failed_states():
    return {"unknown", "failed", "skipped", "cancelled", "failed"}


def get_id(
        awaitable: typing.Union[CreateEmbedJobResponse, DatasetsCreateResponse, EmbedJob, DatasetsGetResponse]):
    return getattr(awaitable, "job_id", None) or getattr(awaitable, "id", None) or getattr(
        getattr(awaitable, "dataset", None), "id", None)


def get_validation_status(awaitable: typing.Union[EmbedJob, DatasetsGetResponse]):
    return getattr(awaitable, "status", None) or getattr(getattr(awaitable, "dataset", None), "validation_status", None)


def get_job(cohere: typing.Any,
            awaitable: typing.Union[CreateEmbedJobResponse, DatasetsCreateResponse, EmbedJob, DatasetsGetResponse]) -> \
typing.Union[
    EmbedJob, DatasetsGetResponse]:
    if awaitable.__class__.__name__ == "EmbedJob" or awaitable.__class__.__name__ == "CreateEmbedJobResponse":
        return cohere.embed_jobs.get(id=get_id(awaitable))
    elif awaitable.__class__.__name__ == "DatasetsGetResponse" or awaitable.__class__.__name__ == "DatasetsCreateResponse":
        return cohere.datasets.get(id=get_id(awaitable))
    else:
        raise ValueError(f"Unexpected awaitable type {awaitable}")


async def async_get_job(cohere: typing.Any, awaitable: typing.Union[CreateEmbedJobResponse, DatasetsCreateResponse]) -> \
        typing.Union[
            EmbedJob, DatasetsGetResponse]:
    if awaitable.__class__.__name__ == "EmbedJob" or awaitable.__class__.__name__ == "CreateEmbedJobResponse":
        return await cohere.embed_jobs.get(id=get_id(awaitable))
    elif awaitable.__class__.__name__ == "DatasetsGetResponse" or awaitable.__class__.__name__ == "DatasetsCreateResponse":
        return await cohere.datasets.get(id=get_id(awaitable))
    else:
        raise ValueError(f"Unexpected awaitable type {awaitable}")


def get_failure_reason(job: typing.Union[EmbedJob, DatasetsGetResponse]) -> Optional[str]:
    if isinstance(job, EmbedJob):
        return f"Embed job {job.job_id} failed with status {job.status}"
    elif isinstance(job, DatasetsGetResponse):
        return f"Dataset creation {job.dataset.validation_status} failed with status {job.dataset.validation_status}"
    return None


@typing.overload
def wait(
        cohere: typing.Any,
        awaitable: CreateEmbedJobResponse,
        timeout: Optional[float] = None,
        interval: float = 10,
) -> EmbedJob:
    ...


@typing.overload
def wait(
        cohere: typing.Any,
        awaitable: DatasetsCreateResponse,
        timeout: Optional[float] = None,
        interval: float = 10,
) -> DatasetsGetResponse:
    ...


def wait(
        cohere: typing.Any,
        awaitable: typing.Union[CreateEmbedJobResponse, DatasetsCreateResponse],
        timeout: Optional[float] = None,
        interval: float = 2,
) -> typing.Union[EmbedJob, DatasetsGetResponse]:
    start_time = time.time()
    terminal_states = get_terminal_states()
    failed_states = get_failed_states()

    job = get_job(cohere, awaitable)
    while get_validation_status(job) not in terminal_states:
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"wait timed out after {timeout} seconds")

        time.sleep(interval)
        print("...")

        job = get_job(cohere, awaitable)

    if get_validation_status(job) in failed_states:
        raise Exception(get_failure_reason(job))

    return job


@typing.overload
async def async_wait(
        cohere: typing.Any,
        awaitable: CreateEmbedJobResponse,
        timeout: Optional[float] = None,
        interval: float = 10,
) -> EmbedJob:
    ...


@typing.overload
async def async_wait(
        cohere: typing.Any,
        awaitable: DatasetsCreateResponse,
        timeout: Optional[float] = None,
        interval: float = 10,
) -> DatasetsGetResponse:
    ...


async def async_wait(
        cohere: typing.Any,
        awaitable: typing.Union[CreateEmbedJobResponse, DatasetsCreateResponse],
        timeout: Optional[float] = None,
        interval: float = 10,
) -> typing.Union[EmbedJob, DatasetsGetResponse]:
    start_time = time.time()
    terminal_states = get_terminal_states()
    failed_states = get_failed_states()

    job = await async_get_job(cohere, awaitable)
    while get_validation_status(job) not in terminal_states:
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"wait timed out after {timeout} seconds")

        await asyncio.sleep(interval)
        print("...")

        job = await async_get_job(cohere, awaitable)

    if get_validation_status(job) in failed_states:
        raise Exception(get_failure_reason(job))

    return job
