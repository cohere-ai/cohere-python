import typing
from cohere.client import Client

# TODO: can we get rid of requests? use std lib instead?
try:
    import requests
except ImportError:
    raise ImportError(
        "Please install the requests package to use local tokenization."
        "\nYou can do so by running `pip install requests`."
    )
try:
    from tokenizers import Tokenizer
except ImportError:
    raise ImportError(
        "Please install the hugging face tokenizers package to use local tokenization."
        "\nYou can do so by running `pip install tokenizers`."
    )

TOKENIZER_CACHE_KEY = "tokenizers"


def tokenizer_cache_key(model_name: str) -> str:
    return f"{TOKENIZER_CACHE_KEY}:{model_name}"


def get_hf_tokenizer(co: Client, model_name: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    tokenizer = co.cache_get(tokenizer_cache_key(model_name))
    if tokenizer is not None:
        return tokenizer

    # ---------- should use co.here API to get the tokenizer URL once the GET model is in place ----------
    # url = f"https://api.cohere.ai/v1/models/{model_name}"
    # response = requests.request(
    #     "GET",
    #     url,
    #     headers={
    #         "Content-Type": "application/json",
    #         "Accept": "application/json",
    #         "Authorization": "Bearer <TOKEN>",
    #     },
    # )
    # response.raise_for_status()
    class FakeRes:
        def json(self):
            return {"tokenizer_url": "https://storage.googleapis.com/cohere-assets/tokenizers/command-v1.json"}
    response = FakeRes()
    # ---------- should use co.here API to get the tokenizer URL once the GET model is in place ----------

    response = requests.request("GET", response.json()["tokenizer_url"])
    response.raise_for_status()
    tokenizer = Tokenizer.from_str(response.text)

    co.cache_set(tokenizer_cache_key(model_name), tokenizer)
    return tokenizer


def local_tokenize(co: Client, model_name: str, text: str) -> typing.List[int]:
    """Tokenizes a given text using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model_name)
    return tokenizer.encode(text).tokens


def local_detokenize(co: Client, model_name: str, tokens: list) -> str:
    """Detokenizes a given list of tokens using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model_name)
    return tokenizer.decode(tokens)
