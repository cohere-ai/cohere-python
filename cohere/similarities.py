from typing import List

class Similarities:
    def __init__(self, similarities: List[float]) -> None:
        self.similarities = similarities

    def __str__(self) -> str:
        return str(self.similarities)
