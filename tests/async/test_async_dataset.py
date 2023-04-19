import asyncio
import io
import json
import time
from typing import Optional

import pytest

from cohere import AsyncClient
from cohere.responses import AsyncDataset


@pytest.mark.asyncio
async def test_async_create_dataset(async_client: AsyncClient):
    job = await async_client.create_dataset(
        name="ci-test",
        data=dummy_file(
            [
                {"text": "this is a text"},
                {"text": "this is another text"},
            ]
        ),
        dataset_type="embed-input",
    )
    job = await async_client.get_dataset(job.id)

    start = time.time()
    while not job.has_terminal_status():
        if time.time() - start > 120:  # 120s timeout
            raise TimeoutError()
        await asyncio.sleep(5)
        job = await async_client.get_dataset(job.id)

    check_result(job, status="validated")


@pytest.mark.asyncio
async def test_async_create_invalid_dataset(async_client: AsyncClient):
    job = await async_client.create_dataset(
        name="ci-test",
        data=dummy_file(
            [
                {"foo": "bar"},
                {"baz": "foz"},
            ]
        ),
        dataset_type="embed-input",
    )
    job = await async_client.get_dataset(job.id)

    start = time.time()
    while not job.has_terminal_status():
        if time.time() - start > 120:  # 120s timeout
            raise TimeoutError()
        await asyncio.sleep(5)
        job = await async_client.get_dataset(job.id)

    check_result(job, status="failed")


@pytest.mark.asyncio
async def test_async_get_dataset(async_client: AsyncClient):
    datasets = await async_client.list_datasets()
    dataset = await async_client.get_dataset(datasets[0].id)
    check_result(dataset)


@pytest.mark.asyncio
async def test_async_list_dataset(async_client: AsyncClient):
    datasets = await async_client.list_datasets()
    assert len(datasets) > 0
    for dataset in datasets:
        check_result(dataset)


def dummy_file(data) -> io.BytesIO:
    final = ""
    for t in data:
        final += json.dumps(t) + "\n"

    binaryData = final.encode()
    vfile = io.BytesIO(binaryData)
    vfile.name = "test.jsonl"
    return vfile


def check_result(dataset: AsyncDataset, status: Optional[str] = None):
    assert dataset.id
    assert dataset.created_at
    assert dataset.dataset_type
    assert dataset.name

    if status is not None:
        assert dataset.validation_status == status

    if status == "validated":
        assert dataset.urls
        for row in dataset.open():
            assert row

    if status == "failed":
        assert dataset.validation_error
