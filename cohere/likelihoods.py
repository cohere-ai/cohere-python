from cohere.response import CohereObject
from typing import List

class Likelihoods(CohereObject):
    def __init__(self, likelihood: float, tokens: List[str], token_likehoods: List[float]) -> None:
        self.likelihood = likelihood
        self.tokens = tokens
        self.token_likehoods = token_likehoods
