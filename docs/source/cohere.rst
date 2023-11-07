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
