![ci badge](https://github.com/cohere-ai/cohere-python/actions/workflows/test.yaml/badge.svg)
![version badge](https://img.shields.io/pypi/v/cohere)
![license badge](https://img.shields.io/github/license/cohere-ai/cohere-python)

# Cohere Python SDK

This package provides functionality developed to simplify interfacing with the [Cohere API](https://docs.cohere.ai/) in Python 3.

## Documentation

* SDK Documentation is hosted on [Read the docs](https://cohere-sdk.readthedocs.io/en/latest/).
  * You can build SDK documentation locally using `cd docs; make clean html`.
* For more details on advanced parameters, you can also consult the [API documentation](https://docs.cohere.ai/reference/about).
* See the [examples](examples/) directory for examples, including  some additional functionality for visualizations in Jupyter notebooks.

## Installation

The package can be installed with `pip`:

```bash
pip install --upgrade cohere
```

Install from source:

```bash
pip install .
```

### Requirements

- Python 3.7+

## Quick Start

To use this library, you must have an API key and specify it as a string when creating the `cohere.Client` object. API keys can be created through the [platform](https://os.cohere.ai). This is a basic example of the creating the client and using the `generate` endpoint.

```python
import cohere

# initialize the Cohere Client with an API Key
co = cohere.Client('YOUR_API_KEY')

# generate a prediction for a prompt
prediction = co.generate(
            model='large',
            prompt='co:here',
            max_tokens=10)

# print the predicted text
print('prediction: {}'.format(prediction.generations[0].text))
```

There is also an asyncio compatible client called `cohere.AsyncClient` with an equivalent interface. Consult the [SDK Docs](https://cohere-sdk.readthedocs.io/en/latest/) for more details.

## Versioning

Each SDK release is only compatible with the latest version of the Cohere API at the time of release. To use the SDK with an older API version, you need to download a version of the SDK tied to the API version you want. Look at the [Changelog](https://github.com/cohere-ai/cohere-python/blob/main/CHANGELOG.md) to see which SDK version to download.


## Endpoints

For a full breakdown of endpoints and arguments, please consult the [SDK Docs](https://cohere-sdk.readthedocs.io/en/latest/) and [Cohere API Docs](https://docs.cohere.ai/).

| Cohere Endpoint  | Function             |
| ---------------- | -------------------- |
| /generate        | co.generate()        |
| /embed           | co.embed()           |
| /classify        | co.classify()        |
| /tokenize        | co.tokenize()        |
| /detokenize      | co.detokenize()      |
| /detect-language | co.detect_language() |

## Models

When you call Cohere's APIs we decide on a good default model for your use-case behind the scenes. The default model is great to get you started, but in production environments we recommend that you specify the model size yourself via the `model` parameter. Learn more about the available models here(https://os.cohere.ai)

## Responses

All of the endpoint functions will return a Cohere object corresponding to the endpoint (e.g. for generation, it would be `Generation`). The responses can be found as instance variables of the object (e.g. generation would be `Generation.text`). The names of these instance variables and a detailed breakdown of the response body can be found in the [SDK Docs](https://cohere-sdk.readthedocs.io/en/latest/) and [Cohere Docs](https://docs.cohere.ai/). Printing the Cohere response object itself will display an organized view of the instance variables.

## Exceptions

Unsuccessful API calls from the SDK will raise an exception. Please see the documentation's page on [errors](https://docs.cohere.ai/errors-reference) for more information about what the errors mean.

## Contributing

To set up a development environment, run:

```
poetry shell    # any time you want to run code or tests
poetry install  # install and update dependencies in your environment, the first time
```

In addition, to ensure your code is formatted correctly, install pre-commit hooks using:

```bash
pre-commit install
```

You can run tests locally using:
```
python -m pytest
```

You can configure a different base url with:
```bash
CO_API_URL="https://localhost:8050" python3 foo.py
```
or
```python
cohere.COHERE_API_URL = "https://localhost:8050" # Place before client initilization
```
