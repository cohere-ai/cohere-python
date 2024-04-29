import asyncio
import logging
import typing

import requests
from tokenizers import Tokenizer  # type: ignore

if typing.TYPE_CHECKING:
    from cohere.client import AsyncClient, Client

TOKENIZER_CACHE_KEY = "tokenizers"
logger = logging.getLogger(__name__)


def tokenizer_cache_key(model: str) -> str:
    return f"{TOKENIZER_CACHE_KEY}:{model}"


def get_hf_tokenizer(co: "Client", model: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    tokenizer = co._cache_get(tokenizer_cache_key(model))
    if tokenizer is not None:
        return tokenizer
    tokenizer_url = co.models.get(model).tokenizer_url
    if not tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model}")

    size = int(typing.cast(int, requests.head(tokenizer_url).headers.get("Content-Length")))
    size_mb = round(size / 1024 / 1024, 2)
    logger.info(f"Downloading tokenizer for model {model}. Size is {size_mb} MBs.")
    response = requests.get(tokenizer_url)
    tokenizer = Tokenizer.from_str(response.text)

    co._cache_set(tokenizer_cache_key(model), tokenizer)
    return tokenizer


def local_tokenize(co: "Client", model: str, text: str) -> typing.List[int]:
    """Encodes a given text using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model)
    return tokenizer.encode(text, add_special_tokens=False).ids


def local_detokenize(co: "Client", model: str, tokens: typing.Sequence[int]) -> str:
    """Decodes a given list of tokens using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model)
    return tokenizer.decode(tokens)


async def async_get_hf_tokenizer(co: "AsyncClient", model: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""

    tokenizer = co._cache_get(tokenizer_cache_key(model))
    if tokenizer is not None:
        return tokenizer
    tokenizer_url = (await co.models.get(model)).tokenizer_url
    if not tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model}")

    size = int(typing.cast(int, requests.head(tokenizer_url).headers.get("Content-Length")))
    size_mb = round(size / 1024 / 1024, 2)
    logger.info(f"Downloading tokenizer for model {model}. Size is {size_mb} MBs.")
    response = await asyncio.get_event_loop().run_in_executor(None, requests.get, tokenizer_url)
    tokenizer = Tokenizer.from_str(response.text)

    co._cache_set(tokenizer_cache_key(model), tokenizer)
    return tokenizer


async def async_local_tokenize(co: "AsyncClient", model: str, text: str) -> typing.List[int]:
    """Encodes a given text using a local tokenizer."""
    tokenizer = await async_get_hf_tokenizer(co, model)
    return tokenizer.encode(text, add_special_tokens=False).ids


async def async_local_detokenize(co: "AsyncClient", model: str, tokens: typing.Sequence[int]) -> str:
    """Decodes a given list of tokens using a local tokenizer."""
    tokenizer = await async_get_hf_tokenizer(co, model)
    return tokenizer.decode(tokens)
