from cohere.response import CohereResponse
from typing import List, Dict

class Generation(CohereResponse):
    def __init__(self, text: str, token_likelihoods: List[Dict], return_likelihoods: str) -> None:
        self.text = text
        self.token_likelihoods = token_likelihoods
        self.return_likelihoods = return_likelihoods
