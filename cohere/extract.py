from cohere.response import CohereObject
from typing import List


class ExtractEntity:
    '''
    ExtractEntity represents a single entity extracted from a text. An entity has a
    type and a value. For the text "I am a plumber", an extracted entity could be
    of type "profession" with the value "plumber".
    '''

    def __init__(self, type: str, value: str) -> None:
        self.type = type
        self.value = value

    def toDict(self) -> dict:
        return {"type": self.type, "value": self.value}

    def __str__(self) -> str:
        return f"{self.type}: {self.value}"


class ExtractExample:
    '''
    ExtractExample represents one of the examples provided to the model. An example
    contains the text of the example and a list of entities extracted from the text.

    >>> example = ExtractExample("I am a plumber", [ExtractEntity("profession", "plumber")])
    >>> example = ExtractExample("Joe is a teacher", [
            ExtractEntity("name", "Joe"), ExtractEntity("profession", "teacher")
        ])
    '''

    def __init__(self, text: str, entities: List[ExtractEntity]) -> None:
        self.text = text
        self.entities = entities

    def toDict(self):
        return {"text": self.text, "entities": [entity.toDict() for entity in self.entities]}

    def __str__(self) -> str:
        return f"{self.text}\n\t{self.entities}"


class Extraction:
    '''
    Represents the results of extracting entities from a single text input. An extraction
    contains the text input, the list of entities extracted from the text, and the id of the
    extraction.
    '''

    def __init__(self, id: str, text: str, entities: List[ExtractEntity]) -> None:
        self.id = id
        self.text = text
        self.entities = entities

    def __str__(self) -> str:
        return f"{self.id}: {self.text}\n\t{self.entities}"


class Extractions(CohereObject):
    '''
    Represents the return value of calling the Extract API. An Extractions object contains
    a list of of Extraction objects, one per text input.
    '''

    def __init__(self, extractions: List[Extraction]) -> None:
        self.extractions = extractions
        self.iterator = iter(extractions)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return len(self.extractions)

    def __str__(self) -> str:
        return f"{self.extractions}"
