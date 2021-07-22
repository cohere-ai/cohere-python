from cohere.response import CohereResponse
from typing import List, Dict

class Likelihoods(CohereResponse):
    def __init__(self, likelihood: float, token_likelihoods: List[Dict]) -> None:
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods
