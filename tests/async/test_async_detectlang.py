import pytest

@pytest.mark.asyncio
async def test_detectlang(async_client):
    res = await async_client.detect_language(["Hello world!", "Привет Мир!"])
    languages = res.results
    assert languages[0].language_code ==  "en"
    assert languages[1].language_code ==  "ru"