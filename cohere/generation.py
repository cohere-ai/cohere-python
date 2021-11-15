from cohere.response import CohereResponse
from typing import List, Dict

class Generation(CohereResponse):
    def __init__(self, texts: List[str], token_likelihoods: List[List[Dict]], return_likelihoods: str) -> None:
        self.texts = texts
        self.token_likelihoods = token_likelihoods
        self.return_likelihoods = return_likelihoods
