import asyncio
import os
import time
from typing import Optional

import pytest

from cohere import AsyncClient
from cohere.responses.cluster import ClusterJobResult

INPUT_FILE = "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_100.jsonl"
IN_CI = os.getenv("CI", "").lower() in ["true", "1"]


@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI, reason="can timeout during high load")
async def test_async_create_cluster_job(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )
    job = await async_client.get_cluster_job(create_res.job_id)
    start = time.time()

    while not job.is_final_state:
        if time.time() - start > 60:  # 60s timeout
            raise TimeoutError()
        await asyncio.sleep(5)
        job = await async_client.get_cluster_job(create_res.job_id)

    check_job_result(job, status="complete")


@pytest.mark.asyncio
async def test_async_get_cluster_job(async_client: AsyncClient):
    jobs = await async_client.list_cluster_jobs()
    job = await async_client.get_cluster_job(jobs[0].job_id)
    check_job_result(job)


@pytest.mark.asyncio
async def test_async_list_cluster_jobs(async_client: AsyncClient):
    jobs = await async_client.list_cluster_jobs()
    assert len(jobs) > 0
    for job in jobs:
        check_job_result(job)


@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI, reason="can timeout during high load")
async def test_async_wait_for_cluster_job_succeeds(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    job = await async_client.wait_for_cluster_job(create_res.job_id, timeout=60, interval=5)
    check_job_result(job, status="complete")


@pytest.mark.asyncio
async def test_async_wait_for_cluster_job_times_out(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    with pytest.raises(TimeoutError):
        await async_client.wait_for_cluster_job(create_res.job_id, timeout=1, interval=0.5)


@pytest.mark.asyncio
@pytest.mark.skip
async def test_async_job_wait_method_succeeds(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    job = await create_res.wait(timeout=60, interval=5)
    check_job_result(job, status="complete")


@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI, reason="can timeout during high load")
async def test_async_job_wait_method_times_out(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    with pytest.raises(TimeoutError):
        await create_res.wait(timeout=1, interval=0.5)


def check_job_result(job: ClusterJobResult, status: Optional[str] = None):
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
