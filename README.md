# Cohere Python SDK

This package provides functionality developed to simplify interfacing with the [cohere.ai](https://cohere.ai) natural language API in Python 3.

## Documentation

See the [API's documentation](https://docs.cohere.ai/). 

Also see some code examples for the SDK [here](https://github.com/cohere-ai/cohere-python/blob/main/sanity-test.py).

## Installation

If you want the package, you can install it through `pip`:

```bash
pip install --upgrade cohere
```

Install from source:

```bash
python setup.py install
```

### Requirements
- Python 3.6+

## Usage

```python
import cohere

# initialize the Cohere Client with an API Key
co = cohere.Client('YOUR_API_KEY')

# generate a prediction for a prompt 
prediction = co.generate(
            model="baseline-shrimp",
            prompt="co:here",
            max_tokens=10)
            
# print the predicted text          
print('prediction: {}'.format(prediction.text))
```

More usage examples can be found [here](https://github.com/cohere-ai/cohere-python/blob/main/sanity-test.py).

## Endpoints
For a full breakdown of endpoints and arguments, please consult the [Cohere Docs](https://docs.cohere.ai/).

Cohere Endpoint | Function
----- | -----
/generate  | co.generate()
/similarity | co.similarity()
/choose-best | co.choose_best()
/embed | co.embed()
/likelihood | co.likelihood()

## Models
To view an up-to-date list of available models please consult the [Cohere CLI](https://docs.cohere.ai/command/). To get started try out `baseline-shrimp` or `baseline-seal`.

## Responses
All of the endpoint functions will return some Cohere object (e.g. for generation, it would be `Generation`). The responses can be found as instance variables of the object (e.g. for generation, it would be `Generation.text`). The names of these instance variables and a detailed breakdown of the response body can be found in the [Cohere Docs](https://docs.cohere.ai/).

## Exceptions

Unsuccessful API calls from the SDK will raise an exception. Please see the documentation's page on [errors](https://docs.cohere.ai/errors-reference) for more information about what the errors mean.

