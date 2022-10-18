from concurrent.futures import Future
from typing import List, Optional

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

    @property
    def length(self):
        return len(self)

    def _init_from_future(self, future: Future):
        self.tokens = AsyncAttribute(future, lambda x: x['tokens'])
        self.token_strings = AsyncAttribute(future, lambda x: x['token_strings'])

    def __len__(self) -> int:
        return len(self.tokens)
