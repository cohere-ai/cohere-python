import pytest

@pytest.mark.asyncio
async def test_async_detectlang(async_client):
    res = await async_client.detect_language(["Hello world!", "Привет Мир!"])
    languages = res.results
    assert languages[0].language_code == "en"
    assert languages[1].language_code == "ru"
    assert res.meta
    assert res.meta["api_version"]
    assert res.meta["api_version"]["version"]
