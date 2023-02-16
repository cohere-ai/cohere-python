from concurrent.futures import Future
from typing import List, Optional

from cohere.responses.base import AsyncAttribute, CohereObject, _df_html

from concurrent.futures import Future
from typing import Optional

from cohere.responses.base import AsyncAttribute, CohereObject
import html
from collections import UserList
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np


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


class BatchedTokens(UserList, CohereObject):
    """Acts a list of Tokens object"""

    # nice jupyter output
    def visualize(self, **kwargs) -> str:
        import pandas as pd

        df = pd.DataFrame.from_dict(
            {f"[{i}].{f}": getattr(t, f) for i, t in enumerate(self) for f in ["token_strings", "tokens"]},
            orient="index",
        )
        return _df_html(df.fillna(""), style={"font-size": "90%"})


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
