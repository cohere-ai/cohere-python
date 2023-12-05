Quickstart
==========

Installation
------------

.. code:: bash

    $ pip install cohere



Instantiate and use the client
------------------------------

There are two clients in the sdk with a common interface

* `Client` is based on the python requests package.
* `AsyncClient` uses the python asyncio interface and aiohttp package.

It is recommended to use `AsyncClient` for performance critical applications with many concurrent calls.

.. code:: python

    from cohere import Client
    co = Client()
    co.generate("Hello, my name is", max_tokens=10)


.. code:: python

    from cohere import AsyncClient
    co = AsyncClient()
    await co.generate("Hello, my name is", max_tokens=10)
    await co.close()  # the AsyncClient client should be closed when done

.. code:: python

    from cohere import AsyncClient
    async with AsyncClient() as co:  # using 'async with' runs check_api_key and closes any sessions automatically
        await co.generate("Hello, my name is", max_tokens=10)


API
===

Client
------
.. autoclass:: cohere.client.Client
    :members:
    :member-order: bysource

AsyncClient
-----------
.. autoclass:: cohere.client_async.AsyncClient
    :members:
    :member-order: bysource

API response objects
--------------------

.. automodule:: cohere.responses.generation
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.tokenize
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.classify
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.embeddings
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.detectlang
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.feedback
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.rerank
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.summarize
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.chat
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.cluster
    :members:
    :member-order: bysource

.. automodule:: cohere.responses.custom_model
    :members:
    :member-order: bysource

CustomModelDataset
--------
.. automodule:: cohere.custom_model_dataset
    :members: CustomModelDataset,InMemoryDataset,CsvDataset,JsonlDataset,TextDataset
    :member-order: bysource

Exceptions
----------
.. automodule:: cohere.error
    :members:
    :member-order: bysource