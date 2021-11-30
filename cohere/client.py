import json
from typing import List
from urllib.parse import urljoin

import requests
from requests import Response

import cohere
from cohere.best_choices import BestChoices
from cohere.embeddings import Embeddings
from cohere.error import CohereError
from cohere.generation import Generations, Generation
from cohere.likelihoods import Likelihoods, TokenLikelihood
from cohere.similarities import Similarities

class Client:
    def __init__(self, api_key: str, version: str = None) -> None:
        self.api_key = api_key
        self.api_url = cohere.COHERE_API_URL
        if version is None:
            self.cohere_version = cohere.COHERE_VERSION
        else:
            self.cohere_version = version

    def generate(
        self, 
        model: str, 
        prompt: str,
        num_generations: int = 1, 
        max_tokens: int = 20, 
        temperature: float = 1.0, 
        k: int = 0, 
        p: float = 0.75,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: List[str] = None,
        return_likelihoods: str = 'NONE'
    ) -> Generation:
        json_body = json.dumps({
            'prompt': prompt,
            'num_generations': num_generations,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods,
        })
        response = self.__request(json_body, cohere.GENERATE_URL, model)

        generations = []
        for gen in response['generations']: 
            token_likelihoods = None
            if 'token_likelihoods' in gen.keys(): 
                token_likelihoods = []
                for l in gen['token_likelihoods']: 
                    likelihood = l['likelihood'] if 'likelihood' in l.keys() else None
                    token_likelihoods.append(TokenLikelihood(l['token'], likelihood))
            generations.append(Generation(gen['text'], token_likelihoods))
        return Generations(generations, return_likelihoods)

    def similarity(self, model: str, anchor: str, targets: List[str]) -> Similarities:
        json_body = json.dumps({
            'anchor': anchor,
            'targets': targets,
        })
        response = self.__request(json_body, cohere.SIMILARITY_URL, model)
        return Similarities(response['similarities'])

    def embed(self, model: str, texts: List[str], truncate: str = 'NONE') -> Embeddings:
        json_body = json.dumps({
            'texts': texts,
            'truncate': truncate,
        })
        response = self.__request(json_body, cohere.EMBED_URL, model)
        return Embeddings(response['embeddings'])

    def choose_best(self, model: str, query: str, options: List[str], mode:  str = '') -> BestChoices:
        json_body = json.dumps({
            'query': query,
            'options': options,
            'mode': mode,
        })
        response = self.__request(json_body, cohere.CHOOSE_BEST_URL, model)
        return BestChoices(response['scores'], response['tokens'], response['token_log_likelihoods'], mode)

    def likelihood(self, model: str, text: str) -> Likelihoods:
        json_body = json.dumps({
            'text': text,
        })
        response = self.__request(json_body, cohere.LIKELIHOOD_URL, model)
        return Likelihoods(response['likelihood'], response['token_likelihoods'])

    def __request(self, json_body, endpoint, model) -> Response:
        headers = {
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json',
            'Request-Source': 'python-sdk',
        }
        if self.cohere_version != '':
            headers['Cohere-Version'] = self.cohere_version

        url = urljoin(self.api_url, model + '/' + endpoint)
        response = requests.request('POST', url, headers=headers, data=json_body)
        try:
            res = json.loads(response.text)
        except:
            raise CohereError(
                message=response.text,
                http_status=response.status_code,
                headers=response.headers)
        if 'message' in res.keys(): # has errors
                raise CohereError(
                    message=res['message'],
                    http_status=response.status_code,
                    headers=response.headers)
        return res
