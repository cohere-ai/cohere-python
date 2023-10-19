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

```
git clone https://github.com/engchina/oci-cohere-python.git; cd oci-cohere-python
```

```
conda create -n oci-cohere-python python==3.10.12 -y
conda activate oci-cohere-python
``` 

```
# poetry export -f requirements.txt -o requirements.txt --without-hashes
pip install -r requirements.txt
```

The package can be installed with `pip`:

Install from source:

```bash
pip install .
```

### For Using OCI Cohere

The value of API_KEY must be set to "oci".

`~/.oci/config` must be set correctly.

Support endpoint and function,

| OCI Cohere Endpoint            | Function             |
|--------------------------------|----------------------|
| 20231130/actions/generateText  | co.generate()        |
| 20231130/actions/embedText     | co.embed()           |
| 20231130/actions/summarizeText | co.summarize()       |


### For Using Cohere

The value of API_KEY must be set to the key your fetch from cohere.com.

### Sync with fork

```
# once only
# git remote -v
# git remote add upstream https://github.com/cohere-ai/cohere-python.git
```

```
git fetch upstream
git checkout main
git merge upstream/main
```

- refer: [configuring-a-remote-repository-for-a-fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork)
- refer: [syncing-a-fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)
