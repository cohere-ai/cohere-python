from cohere.likelihoods import TokenLikelihood
from cohere.response import CohereObject
from typing import List


class Generation(CohereObject):
    def __init__(self, text: str, token_likelihoods: TokenLikelihood) -> None:
        self.text = text
        self.token_likelihoods = token_likelihoods

class Generations(CohereObject):
    def __init__(self, generations: List[Generation], return_likelihoods: str) -> None:
        self.generations = generations
        self.return_likelihoods = return_likelihoods
        self.iterator = iter(generations)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)
