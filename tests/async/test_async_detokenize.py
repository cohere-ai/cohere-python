import pytest


@pytest.mark.asyncio
async def test_detokenize(client):
    detokenized = await client.detokenize([10104, 12221, 974, 514, 34])
    assert detokenized == "detokenize me!"

    detokenizeds = await client.detokenize([[10104, 12221, 974, 514, 34]] * 3)
    assert detokenizeds == ["detokenize me!"] * 3

    detokenized = await client.detokenize([])
    assert detokenized == ""
