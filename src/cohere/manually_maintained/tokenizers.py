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
    tokenizer_url = None
    if isinstance(co, AsyncClient):
        tokenizer_url = (await co.models.get(model)).tokenizer_url
    else:
        tokenizer_url = co.models.get(model).tokenizer_url
    if not tokenizer_url:
        raise ValueError(f"No tokenizer URL found for model {model}")

    size = int(typing.cast(int, requests.head(tokenizer_url).headers.get("Content-Length")))
    size_mb = round(size / 1024 / 1024, 2)
    print(f"Downloading tokenizer for model {model}. Size is {size_mb} MBs.")
    # TODO: make this async compatible, it's blocking. This is fine for now; since it downloads only once.
    resource = requests.get(tokenizer_url)
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
