import json
from cohere.response import CohereObject
from typing import Any, List
from json import JSONEncoder

class ExtractEntity:
    def __init__(self, type: str, value: str) -> None:
        self.type = type
        self.value = value

    def toDict(self):
        return {"type": self.type, "value": self.value}

class ExtractExample: 
    def __init__(self, text: str, entities: List[ExtractEntity]) -> None:
        self.text = text
        self.entities = entities
    
    def toDict(self):
        return {"text": self.text, "entities": [entity.toDict() for entity in self.entities]}

class Extraction: 
    def __init__(self, id: str, text: str, entities: List[ExtractEntity]) -> None:
        self.id = id
        self.text = text
        self.entities = entities

class Extractions(CohereObject):
    def __init__(self, extractions: List[Extraction]) -> None:
        self.extractions = extractions
        self.iterator = iter(extractions)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return len(self.extractions)
