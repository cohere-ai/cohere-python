from cohere.response import CohereObject
from typing import List


class Tokens(CohereObject):
    def __init__(self, tokens: List[int], token_strings: List[str]) -> None:
        self.tokens = tokens
        self.token_strings = token_strings
        self.iterator = iter(tokens)
        self.length = len(tokens)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return self.length
