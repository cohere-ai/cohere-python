import pytest
from utils import get_api_key

import cohere
from cohere.responses.embed_job import EmbedJob


def test_list_get_jobs(co):
    jobs = co.list_embed_jobs()
    if len(jobs) == 0:
        pytest.skip("no jobs to test")  # the account has no jobs to check

    for job in jobs:
        check_job_result(job)
    job = co.get_embed_job(jobs[0].job_id)
    check_job_result(job)


@pytest.fixture
def co() -> cohere.Client:
    return cohere.Client(get_api_key(), check_api_key=False, client_name="test")


def check_job_result(job: EmbedJob):
    assert job.job_id
    assert job.status
    assert job.created_at
    assert job.model
    assert job.truncate
