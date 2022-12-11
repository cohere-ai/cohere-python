import aiohttp
import asyncio
from typing import Any, Optional
from urllib.parse import urljoin

import cohere
from cohere.classify import Classification, Classifications
from cohere.classify import Example as ClassifyExample
from cohere.classify import LabelPrediction
from cohere.detokenize import Detokenization
from cohere.embeddings import Embeddings
from cohere.error import CohereError
from cohere.generation import Generations
from cohere.tokenize import Tokens


class BaseClient:
    """Implements base methods for cohere client."""

    def __init__(self,
                 api_key: str,
                 version: str = None,
                 num_workers: int = 64,
                 request_dict: dict = {},
                 check_api_key: bool = True) -> None:
        self.api_key = api_key
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.cohere_version = version or cohere.COHERE_VERSION
        self.session = self._init_session()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            'Cohere-Version': self.cohere_version,
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json',
            'Request-Source': 'python-sdk'
        }


class AsyncClient(BaseClient):
    """Implements async endpoints for cohere service."""

    def __init__(
        self, api_key: str, version: Optional[str] = None,
        num_workers: int = 64, request_dict: Optional[dict[str, str]] = {},
        check_api_key: bool = True, loop=None):

        self.loop = loop or asyncio.get_event_loop()
        super().__init__(
            api_key, version, num_workers, request_dict, check_api_key)

    @classmethod
    async def create(
        cls, api_key: str, version: Optional[str] = None,
        num_workers: int = 64, request_dict: Optional[dict[str, str]] = {},
        check_api_key: bool = True, loop=None):
        """Creates client and check if api key is valid."""
        self = cls(
            api_key, version, num_workers, request_dict, check_api_key, loop)
        if check_api_key:
            try:
                response = await self.check_api_key()
                if not response['valid']:
                    await self.close_connection()
                    raise CohereError('invalid api key')
                return self
            except Exception as error:
                await self.close_connection()
                raise CohereError from error
        return self

    def _init_session(self) -> aiohttp.ClientSession:
        session = aiohttp.ClientSession(
            loop=self.loop, headers=self._headers)
        return session

    async def close_connection(self):
        if self.session:
            assert self.session
            await self.session.close()

    async def _request(self, endpoint: str, **kwargs):
        """Sends async request and handles response."""
        headers = self._headers
        uri = urljoin(self.api_url, endpoint)
        kwargs.update(**self.request_dict)
        async with self.session.post(uri, headers=headers, **kwargs) as response:
            self.response = response
            return await self._handle_response(response)

    async def _handle_response(self, response: aiohttp.ClientResponse):
        """Internal helper for handling API responses from the Cohere server.
        Raises the appropriate exceptions when necessary; otherwise, returns the
        response.
        """
        if not str(response.status).startswith('2'):
            raise CohereError(
                await response.text(), response.status, response.headers)
        try:
            return await response.json()
        except ValueError:
            txt = await response.text()
            raise CohereError(f'Invalid Response: {txt}')

    async def check_api_key(self) -> dict[str, Any]:
        return await self._request(cohere.CHECK_API_KEY_URL)

    async def batch_generate(self, prompts: list[str], **kwargs
        ) -> list[Generations]:
        coros = []
        for prompt in prompts:
            kwargs['prompt'] = prompt
            coros.append(self.generate(**kwargs))
        return await asyncio.gather(*coros)

    async def generate(self, **kwargs) -> Generations:
        return_likelihoods = kwargs.get('return_likelihoods')
        response = await self._request(cohere.GENERATE_URL, json=kwargs)
        return Generations(return_likelihoods, response)

    async def embed(
        self, texts: list[str], model: str = None, truncate: str = 'NONE'
    ) -> Embeddings:
        """Extracts embeddings for given text prompts."""
        coros = []
        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i:i + self.batch_size]
            data = {
                'model': model, 'texts': texts_batch, 'truncate': truncate}
            coros.append(self._request(cohere.EMBED_URL, json=data))
        responses = await asyncio.gather(*coros)
        embeddings = []
        for response in responses:
            embeddings.extend(response['embeddings'])
        return Embeddings(embeddings)

    async def classify(
        self,
        inputs: list[str] = [],
        model: str = None,
        preset: str = None,
        examples: list[ClassifyExample] = [],
        truncate: str = None) -> Classifications:
        """Classifies input texts based on given examples."""
        examples_dicts: list[dict[str, str]] = []
        for example in examples:
            example_dict = {'text': example.text, 'label': example.label}
            examples_dicts.append(example_dict)
        body = {
            'model': model,
            'preset': preset,
            'inputs': inputs,
            'examples': examples_dicts,
            'truncate': truncate,
        }
        response = await self._request(cohere.CLASSIFY_URL, json=body)
        classifications = []
        for res in response['classifications']:
            labels = {}
            for label, prediction in res['labels'].items():
                labels[label] = LabelPrediction(prediction['confidence'])
            classifications.append(Classification(
                res['input'], res['prediction'], res['confidence'], labels))
        return Classifications(classifications)

    async def batch_tokenize(self, texts: list[str]) -> list[Tokens]:
        coros = [self.tokenize(t) for t in texts]
        return await asyncio.gather(*coros)

    async def tokenize(self, text: str) -> Tokens:
        tokens = await self._request(cohere.TOKENIZE_URL, json={'text': text})
        return Tokens(**tokens)

    async def batch_detokenize(
        self, list_of_tokens: list[list[int]]) -> list[Detokenization]:
        coros = [self.detokenize(t) for t in list_of_tokens]
        return await asyncio.gather(*coros)

    async def detokenize(self, tokens: list[int]) -> Detokenization:
        detokens = await self._request(cohere.DETOKENIZE_URL, json={'tokens': tokens})
        return Detokenization(**detokens)
