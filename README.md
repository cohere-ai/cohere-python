![ci badge](https://github.com/cohere-ai/cohere-python/actions/workflows/test.yaml/badge.svg)
![version badge](https://img.shields.io/pypi/v/cohere)
![license badge](https://img.shields.io/github/license/cohere-ai/cohere-python)

# Cohere Python SDK

This package provides functionality developed to simplify interfacing with the [Cohere API](https://docs.cohere.ai/) in Python 3.

## Documentation

See the [API's documentation](https://docs.cohere.ai/#api-reference). 

## Installation

The package can be installed with `pip`:

```bash
pip install --upgrade cohere
```

Install from source:

```bash
python setup.py install
```

### Requirements
- Python 3.6+

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

## Versioning
To use the SDK with a specific API version, you can specify it when creating the Cohere Client:

```python
import cohere

co = cohere.Client('YOUR_API_KEY', '2021-11-08')
```

## Endpoints
For a full breakdown of endpoints and arguments, please consult the [Cohere Docs](https://docs.cohere.ai/).

Cohere Endpoint | Function
----- | -----
/generate  | co.generate()
/embed | co.embed()
/classify | co.classify()
/tokenize  | co.tokenize()
/detokenize | co.detokenize()

## Models
When you call Cohere's APIs we decide on a good default model for your use-case behind the scenes. The default model is great to get you started, but in production environments we recommend that you specify the model size yourself via the `model` parameter. Learn more about the available models here(https://os.cohere.ai)

## Responses
All of the endpoint functions will return a Cohere object corresponding to the endpoint (e.g. for generation, it would be `Generation`). The responses can be found as instance variables of the object (e.g. generation would be `Generation.text`). The names of these instance variables and a detailed breakdown of the response body can be found in the [Cohere Docs](https://docs.cohere.ai/). Printing the Cohere response object itself will display an organized view of the instance variables.

## Exceptions

Unsuccessful API calls from the SDK will raise an exception. Please see the documentation's page on [errors](https://docs.cohere.ai/errors-reference) for more information about what the errors mean.

