from concurrent.futures import Future
from typing import List

from cohere.response import AsyncAttribute, CohereObject


class TokenLikelihood(CohereObject):
    def __init__(self, token: str, likelihood: float) -> None:
        self.token = token
        self.likelihood = likelihood


class Generation(CohereObject):
    def __init__(self,
                 text: str,
                 likelihood: float,
                 token_likelihoods: List[TokenLikelihood]) -> None:
        self.text = text
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods


class Generations(CohereObject):
    def __init__(self,
                 generations: List[Generation],
                 return_likelihoods: str) -> None:
        self.generations = generations
        self.return_likelihoods = return_likelihoods
        self._iterator = None

    def __iter__(self) -> iter:
        if self._iterator is None:
            self._iterator = iter(self.generations)
        return self._iterator

    def __next__(self) -> next:
        if self._iterator is None:
            self._iterator = iter(self.generations)
        return next(self._iterator)
