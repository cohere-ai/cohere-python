"""
Forked and fixed version of httpx_sse to handle multi-line SSE data fields properly.

This module consolidates all httpx_sse functionality into a single file.
"""
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator, List, Optional, cast

import httpx


class SSEError(httpx.TransportError):
    """Exception raised when SSE processing encounters an error."""
    pass


class ServerSentEvent:
    """Represents a Server-Sent Event."""
    
    def __init__(
        self,
        event: Optional[str] = None,
        data: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> None:
        if not event:
            event = "message"

        if data is None:
            data = ""

        if id is None:
            id = ""

        self._event = event
        self._data = data
        self._id = id
        self._retry = retry

    @property
    def event(self) -> str:
        return self._event

    @property
    def data(self) -> str:
        return self._data

    @property
    def id(self) -> str:
        return self._id

    @property
    def retry(self) -> Optional[int]:
        return self._retry

    def json(self) -> Any:
        return json.loads(self.data)

    def __repr__(self) -> str:
        pieces = [f"event={self.event!r}"]
        if self.data != "":
            pieces.append(f"data={self.data!r}")
        if self.id != "":
            pieces.append(f"id={self.id!r}")
        if self.retry is not None:
            pieces.append(f"retry={self.retry!r}")
        return f"ServerSentEvent({', '.join(pieces)})"


class SSEDecoder:
    """Decoder for Server-Sent Events according to the HTML5 specification."""
    
    def __init__(self) -> None:
        self._event = ""
        self._data: List[str] = []
        self._last_event_id = ""
        self._retry: Optional[int] = None

    def decode(self, line: str) -> Optional[ServerSentEvent]:
        # See: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation  # noqa: E501

        if not line:
            if (
                not self._event
                and not self._data
                and not self._last_event_id
                and self._retry is None
            ):
                return None

            sse = ServerSentEvent(
                event=self._event,
                data="\n".join(self._data),
                id=self._last_event_id,
                retry=self._retry,
            )

            # NOTE: as per the SSE spec, do not reset last_event_id.
            self._event = ""
            self._data = []
            self._retry = None

            return sse

        if line.startswith(":"):
            return None

        fieldname, _, value = line.partition(":")

        if value.startswith(" "):
            value = value[1:]

        if fieldname == "event":
            self._event = value
        elif fieldname == "data":
            self._data.append(value)
        elif fieldname == "id":
            if "\0" in value:
                pass
            else:
                self._last_event_id = value
        elif fieldname == "retry":
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass
        else:
            pass  # Field is ignored.

        return None


class EventSource:
    """EventSource for handling Server-Sent Events from HTTP responses."""
    
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def _check_content_type(self) -> None:
        content_type = self._response.headers.get("content-type", "").partition(";")[0]
        if "text/event-stream" not in content_type:
            raise SSEError(
                "Expected response header Content-Type to contain 'text/event-stream', "
                f"got {content_type!r}"
            )

    @property
    def response(self) -> httpx.Response:
        return self._response

    def iter_sse(self) -> Iterator[ServerSentEvent]:
        self._check_content_type()
        decoder = SSEDecoder()
        
        # Process the raw stream instead of using iter_lines() which may truncate
        # Read the entire response as bytes and process it manually
        raw_data = b""
        for chunk in self._response.iter_bytes():
            raw_data += chunk
        
        # Convert to string and split on newlines manually
        text_data = raw_data.decode('utf-8', errors='replace')
        lines = text_data.split('\n')
        
        for line in lines:
            line = line.rstrip("\r")
            sse = decoder.decode(line)
            if sse is not None:
                yield sse

    async def aiter_sse(self) -> AsyncGenerator[ServerSentEvent, None]:
        self._check_content_type()
        decoder = SSEDecoder()
        lines = cast(AsyncGenerator[str, None], self._response.aiter_lines())
        try:
            async for line in lines:
                line = line.rstrip("\n")
                sse = decoder.decode(line)
                if sse is not None:
                    yield sse
        finally:
            await lines.aclose()


@contextmanager
def connect_sse(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> Iterator[EventSource]:
    """Context manager for connecting to Server-Sent Events with a synchronous client."""
    headers = kwargs.pop("headers", {})
    headers["Accept"] = "text/event-stream"
    headers["Cache-Control"] = "no-store"

    with client.stream(method, url, headers=headers, **kwargs) as response:
        yield EventSource(response)


@asynccontextmanager
async def aconnect_sse(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> AsyncIterator[EventSource]:
    """Async context manager for connecting to Server-Sent Events with an async client."""
    headers = kwargs.pop("headers", {})
    headers["Accept"] = "text/event-stream"
    headers["Cache-Control"] = "no-store"

    async with client.stream(method, url, headers=headers, **kwargs) as response:
        yield EventSource(response)


# Public API
__all__ = ["EventSource", "connect_sse", "aconnect_sse", "ServerSentEvent", "SSEError"]