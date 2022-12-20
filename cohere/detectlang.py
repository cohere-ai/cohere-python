from cohere.response import CohereObject


class Language(CohereObject):

    def __init__(self, code: str, name: str, confidence: float):
        self.language_code = code
        self.language_name = name
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"Language<language_code: \"{self.language_code}\", "
            f"language_name: \"{self.language_name}\", confidence: {self.confidence}>"
        )


class DetectLanguageResponse:

    def __init__(self, results: list[Language]):
        self.results = results
