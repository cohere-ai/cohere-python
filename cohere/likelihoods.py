from cohere.response import CohereResponse
from typing import List, Dict


class TokenLikelihood: 
    def __init__(self, token: str, likelihood: float) -> None:
        self.token = token
        self.likelihood = likelihood 

class Likelihoods(CohereResponse):
    def __init__(self, likelihood: float, token_likelihoods: List[TokenLikelihood]) -> None:
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods
