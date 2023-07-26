import pytest
from utils import get_api_key, in_ci

import cohere
from cohere.responses.embed_job import EmbedJob

BAD_DATASET_ID = "bad-id"


@pytest.mark.skipif(in_ci(), reason="can timeout during high load")
def test_create_job(co):
    create_res = co.create_embed_job(input_dataset=BAD_DATASET_ID)
    job = create_res.wait(timeout=60, interval=5)
    assert job.status == "failed"
    check_job_result(job)


def test_list_jobs(co):
    jobs = co.list_embed_jobs()
    assert len(jobs) > 0
    for job in jobs:
        check_job_result(job)


def test_get_job(co):
    jobs = co.list_embed_jobs()
    job = co.get_embed_job(jobs[0].job_id)
    check_job_result(job)


def test_cancel_job(co):
    jobs = co.list_embed_jobs()
    co.cancel_embed_job(jobs[0].job_id)


@pytest.fixture
def co() -> cohere.Client:
    return cohere.Client(get_api_key(), check_api_key=False, client_name="test")


def check_job_result(job: EmbedJob):
    assert job.job_id
    assert job.status
    assert job.created_at
    assert job.model
    assert job.truncate
