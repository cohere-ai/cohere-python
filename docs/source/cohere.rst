Quickstart
==========

Installation
------------

.. code:: bash

    $ pip install cohere



Instantiate a new client
------------------------

.. code:: python

    >>> from cohere import Client
    >>> co = Client()
    >>> co.generate("Hello, my name is",max_tokens=10)




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
