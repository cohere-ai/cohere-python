from concurrent.futures import Future
from typing import Iterator, List, Optional

from cohere.response import AsyncAttribute, CohereObject


class Tokens(CohereObject):

    def __init__(self,
                 tokens: Optional[List[int]] = None,
                 token_strings: Optional[List[str]] = None,
                 *,
                 _future: Optional[Future] = None) -> None:
        if _future is not None:
            self._init_from_future(_future)
        else:
            assert tokens is not None
            assert token_strings is not None
            self.tokens = tokens
            self.token_strings = token_strings
        self.iterator = iter(self.tokens)
        self.length = len(self.tokens)

    def _init_from_future(self, future: Future):
        self.tokens = AsyncAttribute(future, lambda x: x['tokens'])
        self.token_strings = AsyncAttribute(future, lambda x: x['token_strings'])

    def __iter__(self) -> Iterator:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return self.length
