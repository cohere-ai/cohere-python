from cohere.response import CohereObject


class Summary(CohereObject):
    def __init__(self, text: str, ratio: float) -> None:
        self.text = text
