# Cohere Python SDK

![](banner.png)

[![version badge](https://img.shields.io/pypi/v/cohere)](https://pypi.org/project/cohere/)
![license badge](https://img.shields.io/github/license/cohere-ai/cohere-python)
[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-SDK%20generated%20by%20Fern-brightgreen)](https://github.com/fern-api/fern)

---

## ⚠️ Custom Modifications (Internal Fork)

**This is a modified version of the Cohere Python SDK with the following changes:**

### Async Client Migration (httpx → aiohttp)
- **Date:** February 2026
- **Reason:** Resolves `httpx.ConnectError: All connection attempts failed` issues
- **Scope:** Async clients only (`AsyncClient`, `AsyncClientV2`) - sync clients unchanged

### Modified Files:
- `src/cohere/core/http_client.py` - AsyncHttpClient migrated to aiohttp
- `src/cohere/core/client_wrapper.py` - AsyncClientWrapper updated
- `src/cohere/base_client.py` - AsyncBaseCohere initialization
- `src/cohere/core/http_response.py` - AsyncHttpResponse compatibility
- `src/cohere/core/http_sse/_api.py` - SSE streaming with aiohttp
- `src/cohere/core/http_sse/_exceptions.py` - Exception compatibility
- `src/cohere/core/file.py` - FormData support for aiohttp
- `pyproject.toml` - Added aiohttp dependency

### Testing:
- All async operations verified working (see `test_async_client.py`)
- 8/8 test suite passing: chat, streaming, SSE, embed, concurrent requests, error handling

### Important Notes:
- **Fern-generated code modified:** Changes will be overwritten if Fern regenerates
- **Version pinned:** Stay on 5.20.5 base until migration is upstreamed
- **Backward compatible:** Sync clients (`Client`, `ClientV2`) continue using httpx
- **Production ready:** All async functionality tested and working

**To use:** Install with `uv sync` in this directory

---

The Cohere Python SDK allows access to Cohere models across many different platforms: the cohere platform, AWS (Bedrock, Sagemaker), Azure, GCP and Oracle OCI. For a full list of support and snippets, please take a look at the [SDK support docs page](https://docs.cohere.com/docs/cohere-works-everywhere).

## Documentation

Cohere documentation and API reference is available [here](https://docs.cohere.com/).

## Installation

```
pip install cohere
```

## Usage

```Python
import cohere

co = cohere.ClientV2()

response = co.chat(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "hello world!"}],
)

print(response)
```

> [!TIP]
> You can set a system environment variable `CO_API_KEY` to avoid writing your api key within your code, e.g. add `export CO_API_KEY=theapikeyforyouraccount`
> in your ~/.zshrc or ~/.bashrc, open a new terminal, then code calling `cohere.Client()` will read this key.


## Streaming

The SDK supports streaming endpoints. To take advantage of this feature for chat,
use `chat_stream`.

```Python
import cohere

co = cohere.ClientV2()

response = co.chat_stream(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "hello world!"}],
)

for event in response:
    if event.type == "content-delta":
        print(event.delta.message.content.text, end="")
```

## Contributing

While we value open-source contributions to this SDK, the code is generated programmatically. Additions made directly would have to be moved over to our generation code, otherwise they would be overwritten upon the next generated release. Feel free to open a PR as a proof of concept, but know that we will not be able to merge it as-is. We suggest opening an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
