from typing import Iterator, List

from cohere.response import CohereObject


class Transcripts(CohereObject):

    def __init__(self, texts: List[str]) -> None:
        self.texts = texts

    def __iter__(self) -> Iterator:
        return iter(self.texts)
