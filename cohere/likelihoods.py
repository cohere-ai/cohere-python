from cohere.response import CohereObject
from typing import List


class TokenLikelihood(CohereObject): 
    def __init__(self, token: str, likelihood: float) -> None:
        self.token = token
        self.likelihood = likelihood 

class Likelihoods(CohereObject):
    def __init__(self, likelihood: float, token_likelihoods: List[TokenLikelihood]) -> None:
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods
