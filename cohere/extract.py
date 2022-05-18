from cohere.response import CohereObject
from typing import List


class Entity:
    '''
    Entity represents a single extracted entity from a text. An entity has a
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

    def __repr__(self) -> str:
        return str(self.toDict())

    def __eq__(self, other) -> bool:
        return self.type == other.type and self.value == other.value


class Example:
    '''
    Example represents a sample extraction from a text, to be provided to the model. An Example
    contains the input text and a list of entities extracted from the text.

    >>> example = Example("I am a plumber", [Entity("profession", "plumber")])
    >>> example = Example("Joe is a teacher", [
            Entity("name", "Joe"), Entity("profession", "teacher")
        ])
    '''

    def __init__(self, text: str, entities: List[Entity]) -> None:
        self.text = text
        self.entities = entities

    def toDict(self):
        return {"text": self.text, "entities": [entity.toDict() for entity in self.entities]}

    def __str__(self) -> str:
        return f"{self.text} -> {self.entities}"

    def __repr__(self) -> str:
        return str(self.toDict())


class Extraction:
    '''
    Represents the result of extracting entities from a single text input. An extraction
    contains the text input, the list of entities extracted from the text, and the id of the
    extraction.
    '''

    def __init__(self, id: str, text: str, entities: List[Entity]) -> None:
        self.id = id
        self.text = text
        self.entities = entities

    def __repr__(self) -> str:
        return str(self.toDict())

    def toDict(self) -> dict:
        return {"id": self.id, "text": self.text, "entities": [entity.toDict() for entity in self.entities]}


class Extractions(CohereObject):
    '''
    Represents the main response of calling the Extract API. An Extractions is iterable and
    contains a list of of Extraction objects, one per text input.
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

    def __getitem__(self, index: int) -> Extraction:
        return self.extractions[index]
