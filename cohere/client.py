import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List
from urllib.parse import urljoin

import requests
from requests import Response

import cohere
from cohere.classify import Classification, Classifications, Confidence, LabelPrediction
from cohere.classify import Example as ClassifyExample
from cohere.embeddings import Embeddings
from cohere.error import CohereError
from cohere.extract import Entity
from cohere.extract import Example as ExtractExample
from cohere.extract import Extraction, Extractions
from cohere.generation import Generation, Generations, TokenLikelihood
from cohere.moderate import Moderation, Moderations
from cohere.tokenize import Tokens
from cohere.detokenize import Detokenization

use_xhr_client = False
try:
    from js import XMLHttpRequest
    use_xhr_client = True
except ImportError:
    pass


class Client:

    def __init__(self,
                 api_key: str,
                 version: str = None,
                 num_workers: int = 8,
                 request_dict: dict = {},
                 check_api_key: bool = True) -> None:
        self.api_key = api_key
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        if version is None:
            self.cohere_version = cohere.COHERE_VERSION
        else:
            self.cohere_version = version

        if check_api_key:
            try:
                res = self.check_api_key()
                if not res['valid']:
                    raise CohereError('invalid api key')
            except CohereError as e:
                raise CohereError(message=e.message, http_status=e.http_status, headers=e.headers)

    def check_api_key(self) -> Response:
        headers = {
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json',
            'Request-Source': 'python-sdk',
        }
        if self.cohere_version != '':
            headers['Cohere-Version'] = self.cohere_version

        url = urljoin(self.api_url, cohere.CHECK_API_KEY_URL)
        if use_xhr_client:
            response = self.__pyfetch(url, headers, None)
            return response
        else:
            response = requests.request('POST', url, headers=headers)

        try:
            res = json.loads(response.text)
        except Exception:
            raise CohereError(message=response.text, http_status=response.status_code, headers=response.headers)
        if 'message' in res.keys():  # has errors
            raise CohereError(message=res['message'], http_status=response.status_code, headers=response.headers)
        return res

    def generate(
        self,
        prompt: str = None,
        model: str = None,
        preset: str = None,
        num_generations: int = 1,
        max_tokens: int = 20,
        temperature: float = 1.0,
        k: int = 0,
        p: float = 0.75,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: List[str] = None,
        return_likelihoods: str = 'NONE',
        truncate: str = None
    ) -> Generations:
        json_body = json.dumps({
            'model': model,
            'prompt': prompt,
            'preset': preset,
            'num_generations': num_generations,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods,
            'truncate': truncate,
        })
        response = self.__request(json_body, cohere.GENERATE_URL)

        generations: List[Generation] = []
        for gen in response['generations']:
            likelihood = None
            token_likelihoods = None
            if return_likelihoods == 'GENERATION' or return_likelihoods == 'ALL':
                likelihood = gen['likelihood']
            if 'token_likelihoods' in gen.keys():
                token_likelihoods = []
                for likelihoods in gen['token_likelihoods']:
                    token_likelihood = likelihoods['likelihood'] if 'likelihood' in likelihoods.keys() else None
                    token_likelihoods.append(TokenLikelihood(likelihoods['token'], token_likelihood))
            generations.append(Generation(gen['text'], likelihood, token_likelihoods))
        return Generations(generations, return_likelihoods)

    def embed(self, texts: List[str], model: str = None, truncate: str = 'NONE') -> Embeddings:
        responses = []
        json_bodys = []
        request_futures = []
        num_batch = int(math.ceil(len(texts) / self.batch_size))
        embed_url_stacked = [cohere.EMBED_URL] * num_batch

        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i:i + self.batch_size]
            json_bodys.append(json.dumps({
                'model': model,
                'texts': texts_batch,
                'truncate': truncate,
            }))
        if use_xhr_client:
            for json_body in json_bodys:
                response = self.__request(json_body, cohere.EMBED_URL)
                responses.append(response['embeddings'])
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i in executor.map(self.__request, json_bodys, embed_url_stacked):
                    request_futures.append(i)
            for result in request_futures:
                responses.extend(result['embeddings'])

        return Embeddings(responses)

    def classify(
        self,
        inputs: List[str] = [],
        model: str = None,
        preset: str = None,
        examples: List[ClassifyExample] = [],
        taskDescription: str = '',
        outputIndicator: str = '',
        truncate: str = None
    ) -> Classifications:
        examples_dicts: list[dict[str, str]] = []
        for example in examples:
            example_dict = {'text': example.text, 'label': example.label}
            examples_dicts.append(example_dict)

        json_body = json.dumps({
            'model': model,
            'preset': preset,
            'inputs': inputs,
            'examples': examples_dicts,
            'taskDescription': taskDescription,
            'outputIndicator': outputIndicator,
            'truncate': truncate,
        })
        response = self.__request(json_body, cohere.CLASSIFY_URL)

        classifications = []
        for res in response['classifications']:
            labelObj = {}
            confidenceObj = []
            for i in range(len(res['confidences'])):
                confidenceObj.append(Confidence(res['confidences'][i]['option'], res['confidences'][i]['confidence']))
            for label, prediction in res['labels'].items():
                labelObj[label] = LabelPrediction(prediction['confidence'])
            classifications.append(Classification(res['input'], res['prediction'], confidenceObj, labelObj))

        return Classifications(classifications)

    def moderate(self, inputs: List[str], model: str = None, truncate: str = 'NONE') -> Moderations:
        json_body = json.dumps({
            'model': model,
            'inputs': inputs,
            'truncate': truncate,
        })
        response = self.__request(json_body, cohere.MODERATE_URL)

        moderations = []
        for res in response['results']:
            moderations.append(
                Moderation(res['profanity'], res['hate_speech'], res['violence'], res['self_harm'], res['sexual'],
                           res['sexual_non_consensual'], res['spam']))

        return Moderations(moderations=moderations)

    def unstable_extract(self, examples: List[ExtractExample], texts: List[str]) -> Extractions:
        '''
        Makes a request to the Cohere API to extract entities from a list of texts.
        Takes in a list of cohere.extract.Example objects to specify the entities to extract.
        Returns an cohere.extract.Extractions object containing extractions per text.
        '''

        json_body = json.dumps({
            'texts': texts,
            'examples': [ex.toDict() for ex in examples],
        })
        response = self.__request(json_body, cohere.EXTRACT_URL)
        extractions = []

        for res in response['results']:
            extraction = Extraction(**res)
            extraction.entities = []
            for entity in res['entities']:
                extraction.entities.append(Entity(**entity))

            extractions.append(extraction)

        return Extractions(extractions)

    def tokenize(self, text: str) -> Tokens:
        json_body = json.dumps({
            'text': text,
        })
        response = self.__request(json_body, cohere.TOKENIZE_URL)
        return Tokens(response['tokens'])

    def detokenize(self, tokens: List[int]) -> Detokenization:
        json_body = json.dumps({
            'tokens': tokens,
        })
        response = self.__request(json_body, cohere.DETOKENIZE_URL)
        return Detokenization(response['text'])

    def __print_warning_msg(self, response: Response):
        if 'X-API-Warning' in response.headers:
            print("\033[93mWarning: {}\n\033[0m".format(response.headers['X-API-Warning']), file=sys.stderr)

    def __pyfetch(self, url, headers, json_body) -> Response:
        req = XMLHttpRequest.new()
        req.open('POST', url, False)
        for key, value in headers.items():
            req.setRequestHeader(key, value)
        try:
            req.send(json_body)
        except Exception:
            raise CohereError(message=req.responseText, http_status=req.status, headers=req.getAllResponseHeaders())
        res = json.loads(req.response)
        if 'message' in res.keys():
            raise CohereError(message=res['message'], http_status=req.status, headers=req.getAllResponseHeaders())
        return res

    def __request(self, json_body, endpoint) -> Any:
        headers = {
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json',
            'Request-Source': 'python-sdk',
        }
        if self.cohere_version != '':
            headers['Cohere-Version'] = self.cohere_version

        url = urljoin(self.api_url, endpoint)
        if use_xhr_client:
            response = self.__pyfetch(url, headers, json_body)
            self.__print_warning_msg(response)
            return response
        else:
            response = requests.request('POST', url, headers=headers, data=json_body, **self.request_dict)
            try:
                res = json.loads(response.text)
            except Exception:
                raise CohereError(message=response.text, http_status=response.status_code, headers=response.headers)
            if 'message' in res.keys():  # has errors
                raise CohereError(message=res['message'], http_status=response.status_code, headers=response.headers)
            self.__print_warning_msg(response)

        return res
