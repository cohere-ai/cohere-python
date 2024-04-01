import requests
import typing
from tokenizers import Tokenizer  # type: ignore

if typing.TYPE_CHECKING:
    from cohere.client import AsyncClient, Client

TOKENIZER_CACHE_KEY = "tokenizers"


def tokenizer_cache_key(model: str) -> str:
    return f"{TOKENIZER_CACHE_KEY}:{model}"


async def get_hf_tokenizer(co: typing.Union["AsyncClient", "Client"], model: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    from cohere.client import AsyncClient

    tokenizer = co._cache_get(tokenizer_cache_key(model))
    if tokenizer is not None:
        return tokenizer

    if isinstance(co, AsyncClient):
        response = await co.models.get(model)
    else:
        response = co.models.get(model)
    if not response.tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model}")

    size = int(typing.cast(int, requests.head(response.tokenizer_url).headers.get("Content-Length")))
    size_mb = round(size / 1024 / 1024, 2)
    print(f"Downloading tokenizer for model {model}. Size is {size_mb} MBs.")
    resource = requests.get(response.tokenizer_url)  # TODO: make this async compatible
    tokenizer = Tokenizer.from_str(resource.text)

    co._cache_set(tokenizer_cache_key(model), tokenizer)
    return tokenizer


async def local_tokenize(co: typing.Union["AsyncClient", "Client"], model: str, text: str) -> typing.List[int]:
    """Encodes a given text using a local tokenizer."""
    tokenizer = await get_hf_tokenizer(co, model)
    return tokenizer.encode(text, add_special_tokens=False).ids


async def local_detokenize(co: typing.Union["AsyncClient", "Client"], model: str, tokens: typing.Sequence[int]) -> str:
    """Decodes a given list of tokens using a local tokenizer."""
    tokenizer = await get_hf_tokenizer(co, model)
    return tokenizer.decode(tokens)
