
import pytest
import time
from cohere.responses.summarize import SummarizeResponse


@pytest.mark.asyncio
async def test_async_cluster_job(async_client):
    create_res = await async_client.create_cluster_job(
        "gs://cohere-dev-central-2/cluster_tests/all_datasets/reddit_500.jsonl",
        min_cluster_size=3,
        threshold=0.5,
    )
    job = await async_client.get_cluster_job(create_res.job_id)
    start = time.time()

    while job.status == 'processing':
        if time.time() - start > 60:  # 60s timeout
            raise TimeoutError()
        time.sleep(5)
        job = await async_client.get_cluster_job(create_res.job_id)

    assert job.status == 'complete'
    assert job.output_clusters_url is not None
