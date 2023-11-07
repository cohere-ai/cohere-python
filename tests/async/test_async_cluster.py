from typing import Optional

import pytest

from cohere import AsyncClient
from cohere.responses.cluster import AsyncClusterJobResult


@pytest.mark.asyncio
async def test_async_list_get_cluster_job(async_client: AsyncClient):
    jobs = await async_client.list_cluster_jobs()
    if len(jobs) == 0:
        pytest.skip("no jobs to test")  # account has no jobs to test

    for job in jobs:
        check_job_result(job)

    job = await async_client.get_cluster_job(jobs[0].job_id)
    check_job_result(job)


def check_job_result(job: AsyncClusterJobResult, status: Optional[str] = None):
    assert job.job_id
    assert job.status
    assert job.meta
    assert job.meta["api_version"]
    assert job.meta["api_version"]["version"]

    if status is not None:
        assert job.status == status

    if status == "complete":
        assert job.output_clusters_url
        assert job.output_outliers_url
        assert job.clusters
