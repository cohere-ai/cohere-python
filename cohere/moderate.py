from typing import List

from cohere.response import CohereObject


class Moderation(CohereObject):

    def __init__(self, profanity: float, hate_speech: float, violence: float, self_harm: float, sexual: float,
                 sexual_non_consensual: float, spam: float) -> None:
        self.profanity = profanity
        self.hate_speech = hate_speech
        self.violence = violence
        self.self_harm = self_harm
        self.sexual = sexual
        self.sexual_non_consensual = sexual_non_consensual
        self.spam = spam


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
