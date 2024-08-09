from .client import Client, AsyncClient
from src.cohere.v2.client import V2Client, AsyncV2Client


class ClientV2(V2Client, Client):
    __init__ = Client.__init__


class AsyncClientV2(AsyncV2Client, AsyncClient):
    __init__ = AsyncClient.__init__
