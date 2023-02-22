# tests consistency between sync and async client and documentation

from cohere import Client, AsyncClient
import inspect


EXPECTED_EXTRA_ASYNC = {'close'}
EXPECTED_DIFFERENT_RETURN_TYPE = {'create_cluster_job'}

def test_consistency_and_docs():
    # Get a list of all non-private functions in the Client class
    client_methods = {name: func for name, func in inspect.getmembers(Client) if not name.startswith("_") and inspect.isfunction(func)}

    # Get a list of all non-private coroutine functions in the AsyncClient class
    async_client_methods = {name: func for name, func in inspect.getmembers(AsyncClient) if not name.startswith("_") and inspect.iscoroutinefunction(func)}
    
    extra_methods = set(async_client_methods.keys()) - set(client_methods.keys()) - EXPECTED_EXTRA_ASYNC
    missing_methods = set(client_methods.keys()) - set(async_client_methods.keys())

    assert extra_methods == set()
    assert missing_methods == set()

    for name, func in client_methods.items():
        assert func.__doc__ is not None, f"Missing documentation for Client.{name}"
        if name in EXPECTED_DIFFERENT_RETURN_TYPE: # async vs sync job type
            assert inspect.signature(func).parameters == inspect.signature(async_client_methods[name]).parameters
        elif not name.startswith('batch_'): # these have return_exceptions
            assert inspect.signature(func) == inspect.signature(async_client_methods[name])
    

