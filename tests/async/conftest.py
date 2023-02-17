import asyncio
import os
import random
import string

import pytest
import pytest_asyncio

from cohere import AsyncClient


@pytest.fixture(scope="session")
def event_loop():  # TODO: intermittent event loop closed thing
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_client() -> AsyncClient:
    api_key = os.getenv("CO_API_KEY")
    assert api_key, "CO_API_KEY environment variable not set"
    client = AsyncClient(api_key, client_name="unittest")
    yield client
    await client.close()



def random_word(length=10):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def random_sentence(num_words):
    return " ".join(random_word() for _ in range(num_words))


@pytest.fixture
def random_texts(num_texts, num_words_per_sentence=50):
    return [random_sentence(num_words_per_sentence) for _ in range(num_texts)]
