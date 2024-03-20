# Cohere Python SDK

![](banner.png)

![version badge](https://img.shields.io/pypi/v/cohere)
![license badge](https://img.shields.io/github/license/cohere-ai/cohere-python)
[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-SDK%20generated%20by%20Fern-brightgreen)](https://github.com/fern-api/fern)

The Cohere Python SDK provides access to the Cohere API from Python.

## ✨🪩✨ Announcing Cohere's new Python SDK ✨🪩✨

We are very excited to publish this brand-new Python SDK. We will continuously update this library with all the latest features in our API. Please create issues where you have feedback so that we can continue to improve the developer experience!

## cohere==5.0.0 Migration Guide

We have created a [migration guide](4.0.0-5.0.0-migration-guide.md) to help you through the process. If you have any questions, please feel free to open an issue and we will respond to you asap.

## Documentation

Cohere documentation and API reference is available [here](https://docs.cohere.com/).

## Installation

```
pip install cohere
```

## Usage

```Python
import cohere

co = cohere.Client(
    api_key="YOUR_API_KEY",
)

chat = co.chat(
    message="hello world!",
    model="command"
)

print(chat)
```

> [!TIP]
> You can set a system environment variable `CO_API_KEY` to avoid writing your api key within your code, e.g. add `export CO_API_KEY=theapikeyforyouraccount`
> in your ~/.zshrc or ~/.bashrc, open a new terminal, then code calling `cohere.Client()` will read this key.


## Streaming

The SDK supports streaming endpoints. To take advantage of this feature for chat,
use `chat_stream`.

```Python
import cohere

co = cohere.Client(
    api_key="YOUR_API_KEY",
)

stream = co.chat_stream(
    message="Tell me a short story"
)

for event in stream:
    if event.event_type == "text-generation":
        print(event.text, end='')
```

## Contributing

While we value open-source contributions to this SDK, the code is generated programmatically. Additions made directly would have to be moved over to our generation code, otherwise they would be overwritten upon the next generated release. Feel free to open a PR as a proof of concept, but know that we will not be able to merge it as-is. We suggest opening an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
