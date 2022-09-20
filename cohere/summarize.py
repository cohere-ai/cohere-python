from cohere.response import CohereObject


class Summary(CohereObject):
    def __init__(self, text: str) -> None:
        self.text = text
