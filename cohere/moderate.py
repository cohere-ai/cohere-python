from typing import List

from cohere.response import CohereObject


class Moderation(CohereObject):

    def __init__(self, benign: float, profanity: float, hate_speech: float, violence: float, self_harm: float,
                 sexual: float, sexual_non_consensual: float, spam: float, information_hazard: float) -> None:
        self.benign = benign
        self.profanity = profanity
        self.hate_speech = hate_speech
        self.violence = violence
        self.self_harm = self_harm
        self.sexual = sexual
        self.sexual_non_consensual = sexual_non_consensual
        self.spam = spam
        self.information_hazard = information_hazard


class Moderations(CohereObject):

    def __init__(self, moderations: List[Moderation]) -> None:
        self.moderations = moderations
        self.iterator = iter(moderations)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)

    def __len__(self) -> int:
        return len(self.moderations)
