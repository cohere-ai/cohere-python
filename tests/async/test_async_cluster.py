import pytest
import asyncio
import time
import os
from cohere import AsyncClient

INPUT_FILE = "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_500.jsonl"
IN_CI = os.getenv('CI', '').lower() in ['true', '1']


@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI,reason="can timeout during high load")
async def test_async_cluster_job(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )
    job = await async_client.get_cluster_job(create_res.job_id)
    start = time.time()

    while job.status == 'processing':
        if time.time() - start > 60:  # 60s timeout
            raise TimeoutError()
        await asyncio.sleep(5)
        job = await async_client.get_cluster_job(create_res.job_id)

    assert job.status == 'complete'
    assert job.output_clusters_url is not None
    assert job.output_outliers_url is not None

@pytest.mark.asyncio
@pytest.mark.skipif(IN_CI,reason="can timeout during high load")
async def test_wait_succeeds(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    job = await async_client.wait_for_cluster_job(create_res.job_id, timeout=60, interval=5)
    assert job.status == 'complete'
    assert job.output_clusters_url is not None
    assert job.output_outliers_url is not None

@pytest.mark.asyncio
async def test_wait_times_out(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    with pytest.raises(TimeoutError):
        await async_client.wait_for_cluster_job(create_res.job_id, timeout=1, interval=0.5)

@pytest.mark.asyncio
@pytest.mark.skip
async def test_handler_wait_succeeds(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    job = await create_res.wait(timeout=60, interval=5)
    assert job.status == 'complete'
    assert job.output_clusters_url is not None
    assert job.output_outliers_url is not None

@pytest.mark.asyncio
async def test_handler_wait_times_out(async_client: AsyncClient):
    create_res = await async_client.create_cluster_job(
        INPUT_FILE,
        min_cluster_size=3,
        threshold=0.5,
    )

    with pytest.raises(TimeoutError):
        await create_res.wait(timeout=1, interval=0.5)
