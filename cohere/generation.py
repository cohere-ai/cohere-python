from cohere.likelihoods import TokenLikelihood
from cohere.response import CohereResponse
from typing import List, Dict


class Generation:
    def __init__(self, text: str, token_likelihoods: TokenLikelihood) -> None:
        self.text = text
        self.token_likelihoods = token_likelihoods

class Generations(CohereResponse):
    def __init__(self, generations: List[Generation], return_likelihoods: str) -> None:
        self.generations = generations
        self.return_likelihoods = return_likelihoods
