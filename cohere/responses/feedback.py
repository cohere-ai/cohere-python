from typing import NamedTuple

GenerateFeedbackResponse = NamedTuple("GenerateFeedbackResponse", [("id", str)])
GeneratePreferenceFeedbackResponse = NamedTuple("GeneratePreferenceFeedbackResponse", [("id", str)])

PreferenceRating = NamedTuple("PreferenceRating", [("request_id", str), ("rating", float), ("generation", str)])
