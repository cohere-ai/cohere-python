import os

import pytest

from cohere import AsyncClient
from cohere.responses.embed_job import EmbedJob

BAD_DATASET_ID = "bad-id"
IN_CI = os.getenv("CI", "").lower() in ["true", "1"]


@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI, reason="can timeout during high load")
async def test_create_job(async_client: AsyncClient):
    create_res = await async_client.create_embed_job(input_dataset=BAD_DATASET_ID)
    job = await create_res.wait(timeout=60, interval=5)
    assert job.status == "failed"
    check_job_result(job)


@pytest.mark.asyncio
async def test_list_jobs(async_client: AsyncClient):
    jobs = await async_client.list_embed_jobs()
    assert len(jobs) > 0
    for job in jobs:
        check_job_result(job)


@pytest.mark.asyncio
async def test_get_job(async_client: AsyncClient):
    jobs = await async_client.list_embed_jobs()
    job = await async_client.get_embed_job(jobs[0].job_id)
    check_job_result(job)


@pytest.mark.asyncio
async def test_cancel_job(async_client: AsyncClient):
    jobs = await async_client.list_embed_jobs()
    await async_client.cancel_embed_job(jobs[0].job_id)


def check_job_result(job: EmbedJob):
    assert job.job_id
    assert job.status
    assert job.created_at
    assert job.model
    assert job.truncate
