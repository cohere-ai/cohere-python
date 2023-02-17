from concurrent.futures import Future
from typing import List, Optional

from cohere.responses.base import CohereObject, _df_html

from concurrent.futures import Future
from typing import Optional

import html
from collections import UserList
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np


class Tokens(CohereObject):

    def __init__(self,
                 tokens: Optional[List[int]] = None,
                 token_strings: Optional[List[str]] = None) -> None:
        assert tokens is not None
        assert token_strings is not None
        self.tokens = tokens
        self.token_strings = token_strings

    @property
    def length(self):
        return len(self)

    def __len__(self) -> int:
        return len(self.tokens)



class Detokenization(CohereObject):

    def __init__(self, text: Optional[str] = None, *, _future: Optional[Future] = None) -> None:
        assert text is not None
        self.text = text

    def __str__(self) -> str:
        return self.text
