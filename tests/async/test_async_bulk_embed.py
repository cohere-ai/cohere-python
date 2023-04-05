import os

import pytest

from cohere import AsyncClient
from cohere.responses.bulk_embed import BulkEmbedJob

BAD_INPUT_FILE = "./local-file.jsonl"
IN_CI = os.getenv("CI", "").lower() in ["true", "1"]


@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI, reason="can timeout during high load")
async def test_create_job(async_client: AsyncClient):
    create_res = await async_client.create_bulk_embed_job(input_file_url=BAD_INPUT_FILE)
    job = await create_res.wait(timeout=60, interval=5)
    assert job.status == "failed"
    check_job_result(job)


@pytest.mark.asyncio
async def test_list_jobs(async_client: AsyncClient):
    jobs = await async_client.list_bulk_embed_jobs()
    assert len(jobs) > 0
    for job in jobs:
        check_job_result(job)


@pytest.mark.asyncio
async def test_get_job(async_client: AsyncClient):
    jobs = await async_client.list_bulk_embed_jobs()
    job = await async_client.get_bulk_embed_job(jobs[0].job_id)
    check_job_result(job)


@pytest.mark.asyncio
async def test_cancel_job(async_client: AsyncClient):
    jobs = await async_client.list_bulk_embed_jobs()
    await async_client.cancel_bulk_embed_job(jobs[0].job_id)


def check_job_result(job: BulkEmbedJob):
    assert job.job_id
    assert job.status
    assert job.created_at
    assert job.input_url
    assert job.model
    assert job.truncate
