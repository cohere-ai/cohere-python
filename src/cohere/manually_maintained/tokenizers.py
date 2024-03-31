import urllib
import typing
from cohere.client import AsyncClient, Client
from tokenizers import Tokenizer

TOKENIZER_CACHE_KEY = "tokenizers"


def tokenizer_cache_key(model_name: str) -> str:
    return f"{TOKENIZER_CACHE_KEY}:{model_name}"


# TODO: will this type hint work if lib is not installed?
def get_hf_tokenizer(co: Client, model_name: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    tokenizer = co._cache_get(tokenizer_cache_key(model_name))
    if tokenizer is not None:
        return tokenizer

    response = co.models.get(model_name)
    if not response.tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model_name}")

    resource = urllib.request.urlopen(response.tokenizer_url)
    tokenizer = Tokenizer.from_str(resource.read().decode("utf-8"))

    co._cache_set(tokenizer_cache_key(model_name), tokenizer)
    return tokenizer


def local_tokenize(co: Client, model_name: str, text: str) -> typing.List[int]:
    """Tokenizes a given text using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model_name)
    return tokenizer.encode(text, add_special_tokens=False).ids


def local_detokenize(co: Client, model_name: str, tokens: typing.List[int]) -> str:
    """Detokenizes a given list of tokens using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model_name)
    return tokenizer.decode(tokens)


async def async_get_hf_tokenizer(co: AsyncClient, model_name: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    tokenizer = co._cache_get(tokenizer_cache_key(model_name))
    if tokenizer is not None:
        return tokenizer

    response = await co.models.get(model_name)
    if not response.tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model_name}")

    resource = urllib.request.urlopen(response.tokenizer_url)
    tokenizer = Tokenizer.from_str(resource.read().decode("utf-8"))

    co._cache_set(tokenizer_cache_key(model_name), tokenizer)
    return tokenizer


async def async_local_tokenize(co: AsyncClient, model_name: str, text: str) -> typing.List[int]:
    """Tokenizes a given text using a local tokenizer."""
    tokenizer = await async_get_hf_tokenizer(co, model_name)
    return tokenizer.encode(text, add_special_tokens=False).ids


async def async_local_detokenize(co: AsyncClient, model_name: str, tokens: typing.List[int]) -> str:
    """Detokenizes a given list of tokens using a local tokenizer."""
    tokenizer = await async_get_hf_tokenizer(co, model_name)
    return tokenizer.decode(tokens)
