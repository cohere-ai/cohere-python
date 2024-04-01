import urllib
import typing
from cohere.client import AsyncClient, Client
from tokenizers import Tokenizer  # type: ignore

TOKENIZER_CACHE_KEY = "tokenizers"


def tokenizer_cache_key(model: str) -> str:
    return f"{TOKENIZER_CACHE_KEY}:{model}"


async def get_hf_tokenizer(co: typing.Union[AsyncClient, Client], model: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    tokenizer = co._cache_get(tokenizer_cache_key(model))
    if tokenizer is not None:
        return tokenizer

    if isinstance(co, AsyncClient):
        response = await co.models.get(model)
    else:
        response = co.models.get(model)
    if not response.tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model}")

    resource = urllib.request.urlopen(response.tokenizer_url)
    tokenizer = Tokenizer.from_str(resource.read().decode("utf-8"))

    co._cache_set(tokenizer_cache_key(model), tokenizer)
    return tokenizer


async def local_tokenize(co: typing.Union[AsyncClient, Client], model: str, text: str) -> typing.List[int]:
    """Encodes a given text using a local tokenizer."""
    tokenizer = await get_hf_tokenizer(co, model)
    return tokenizer.encode(text, add_special_tokens=False).ids


async def local_detokenize(co: typing.Union[AsyncClient, Client], model: str, tokens: typing.Sequence[int]) -> str:
    """Decodes a given list of tokens using a local tokenizer."""
    tokenizer = await get_hf_tokenizer(co, model)
    return tokenizer.decode(tokens)
