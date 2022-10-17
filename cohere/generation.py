from concurrent.futures import Future
from typing import Any, Dict, Iterator, List, Optional, Union

from cohere.response import AsyncAttribute, CohereObject


class TokenLikelihood(CohereObject):

    def __init__(self, token: str, likelihood: float) -> None:
        self.token = token
        self.likelihood = likelihood


class Generation(CohereObject):

    def __init__(self, text: str, likelihood: float, token_likelihoods: List[TokenLikelihood]) -> None:
        self.text = text
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods

    def __str__(self) -> str:
        return str(self.text)


class Generations(CohereObject):

    def __init__(self,
                 response: Optional[Dict[str, Any]] = None,
                 return_likelihoods: Optional[str] = None,
                 *,
                 _future: Optional[Future] = None) -> None:
        self.generations: Union[AsyncAttribute, List[Generation]] = None
        if _future is not None:
            self._init_from_future(_future)
        else:
            assert response is not None
            self.generations = self._generations(response)
            self.return_likelihoods = return_likelihoods
        self._iterator = None

    def _init_from_future(self, future: Future):
        self.generations = AsyncAttribute(future, self._generations)

    def _generations(self, response: Dict[str, Any]) -> List[Generation]:
        generations: List[Generation] = []
        for gen in response['generations']:
            likelihood = None
            if self.return_likelihoods == 'GENERATION' or self.return_likelihoods == 'ALL':
                likelihood = gen['likelihood']
            if 'token_likelihoods' in gen.keys():
                token_likelihoods = []
                for likelihoods in gen['token_likelihoods']:
                    token_likelihood = likelihoods['likelihood'] if 'likelihood' in likelihoods.keys() else None
                    token_likelihoods.append(TokenLikelihood(likelihoods['token'], token_likelihood))
            generations.append(Generation(gen['text'], likelihood, token_likelihoods))

        return generations

    def __str__(self) -> str:
        return str(self.generations)

    def __iter__(self) -> Iterator:
        if self._iterator is None:
            self._iterator = iter(self.generations)
        return self._iterator

    def __next__(self) -> Generation:
        if self._iterator is None:
            self._iterator = iter(self.generations)
        return next(self._iterator)
