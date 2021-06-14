from typing import List, Dict

class Generation:
    def __init__(self, text: str, token_likelihoods: List[Dict], return_likelihoods: str) -> None:
        self.text = text
        self.token_likelihoods = token_likelihoods
        self.return_likelihoods = return_likelihoods
    
    def __str__(self) -> str:
        return self.text
