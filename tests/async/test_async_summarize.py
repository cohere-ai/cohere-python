import pytest

from cohere.responses.summarize import SummarizeResponse

# 1. for oci
MODEL = "cohere.command"


@pytest.mark.asyncio
async def test_summarize(async_client):
    res = await async_client.summarize("".join(f"{i} " for i in range(250)), model=MODEL, )
    assert isinstance(res, SummarizeResponse)
    assert res.meta
    assert res.meta["api_version"]
    assert res.meta["api_version"]["version"]
