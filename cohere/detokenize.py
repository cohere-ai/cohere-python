from cohere.response import CohereObject


class Detokenization(CohereObject):
    def __init__(self, text: str) -> None:
        self.text = text
