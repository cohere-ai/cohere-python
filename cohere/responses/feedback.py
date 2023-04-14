from dataclasses import dataclass
from typing import NamedTuple

GenerateFeedbackResponse = NamedTuple("GenerateFeedbackResponse", [("id", str)])
GeneratePreferenceFeedbackResponse = NamedTuple("GeneratePreferenceFeedbackResponse", [("id", str)])


@dataclass
class PreferenceRating:
    def __init__(self, request_id: str, rating: float, generation: str):
        self.request_id = request_id
        self.rating = rating
        self.generation = generation
