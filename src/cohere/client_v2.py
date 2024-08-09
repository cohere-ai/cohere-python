from .client import Client, AsyncClient
from .v2.client import V2Client, AsyncV2Client


class ClientV2(V2Client, Client):  # type: ignore
    __init__ = Client.__init__  # type: ignore


class AsyncClientV2(AsyncV2Client, AsyncClient):  # type: ignore
    __init__ = AsyncClient.__init__  # type: ignore
