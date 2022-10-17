from concurrent.futures import Future
from typing import Optional

from cohere.response import AsyncAttribute, CohereObject


class Detokenization(CohereObject):

    def __init__(self, text: Optional[str] = None, *, _future: Optional[Future] = None) -> None:
        if _future is not None:
            self._init_from_future(_future)
        else:
            assert text is not None
            self.text = text

    def _init_from_future(self, future: Future):
        self.text = AsyncAttribute(future, lambda x: x['text'])

    def __str__(self) -> str:
        return self.text
