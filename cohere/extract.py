import json
from cohere.response import CohereObject
from typing import List
from json import JSONEncoder

class ExtractEntity:
    def __init__(self, type: str, value: str) -> None:
        self.type = type
        self.value = value
    
class Extraction: 
    def __init__(self, id: str, text: str, entities: List[ExtractEntity]) -> None:
        self.id = id
        self.text = text
        self.entities = entities

class ExtractExample: 
    def __init__(self, text: str, entities: List[ExtractEntity]) -> None:
        self.text = text
        self.entities = entities

class Extract:
    def __init__(self, texts: List[str], examples: List[ExtractExample]) -> None:
        self.texts = texts
        self.examples = examples
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)

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