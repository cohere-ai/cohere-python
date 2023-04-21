from dataclasses import dataclass
from typing import NamedTuple

GenerateFeedbackResponse = NamedTuple("GenerateFeedbackResponse", [("id", str)])
GeneratePreferenceFeedbackResponse = NamedTuple("GeneratePreferenceFeedbackResponse", [("id", str)])


@dataclass
class PreferenceRating:
    request_id: str
    rating: float
    generation: str
