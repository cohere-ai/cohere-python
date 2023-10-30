import pytest

from cohere import AsyncClient
from cohere.responses.embed_job import EmbedJob


@pytest.mark.asyncio
async def test_list_and_get_jobs(async_client: AsyncClient):
    jobs = await async_client.list_embed_jobs()
    if len(jobs) == 0:
        pytest.skip("no jobs to test")  # the account has no jobs to check

    for job in jobs:
        check_job_result(job)
    job = await async_client.get_embed_job(jobs[0].job_id)
    check_job_result(job)


def check_job_result(job: EmbedJob):
    assert job.job_id
    assert job.status
    assert job.created_at
    assert job.model
    assert job.truncate
