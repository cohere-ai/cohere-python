import typing

from httpx import SyncByteStream


class Streamer(SyncByteStream):
    """Wrap an iterator of bytes for httpx streaming responses."""

    lines: typing.Iterator[bytes]

    def __init__(self, lines: typing.Iterator[bytes]):
        self.lines = lines

    def __iter__(self) -> typing.Iterator[bytes]:
        return self.lines
