import urllib
import typing
from cohere.client import Client

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


# TODO: will this type hint work if lib is not installed?
def get_hf_tokenizer(co: Client, model_name: str) -> Tokenizer:
    """Returns a HF tokenizer from a given tokenizer config URL."""
    tokenizer = co._cache_get(tokenizer_cache_key(model_name))
    if tokenizer is not None:
        return tokenizer

    # ---------- should use co.here API to get the tokenizer URL once the GET model is in place ----------
    class FakeRes:
        def json(self):
            return {"tokenizer_url": "https://storage.googleapis.com/cohere-assets/tokenizers/command-v1.json"}

    response = FakeRes()
    # ---------- should use co.here API to get the tokenizer URL once the GET model is in place ----------

    resource = urllib.request.urlopen(response.json()["tokenizer_url"])
    tokenizer = Tokenizer.from_str(resource.read().decode("utf-8"))

    co._cache_set(tokenizer_cache_key(model_name), tokenizer)
    return tokenizer


def local_tokenize(co: Client, model_name: str, text: str) -> typing.List[int]:
    """Tokenizes a given text using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model_name)
    return tokenizer.encode(text, add_special_tokens=False).ids


def local_detokenize(co: Client, model_name: str, tokens: list) -> str:
    """Detokenizes a given list of tokens using a local tokenizer."""
    tokenizer = get_hf_tokenizer(co, model_name)
    return tokenizer.decode(tokens)
