import json
from urllib.parse import urljoin

from typing import List
import requests
from requests import Response

import cohere
from cohere.error import CohereError

from cohere.generation import Generation
from cohere.similarities import Similarities
from cohere.embeddings import Embeddings
from cohere.best_choices import BestChoices
from cohere.likelihoods import Likelihoods


class CohereClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.api_url = cohere.COHERE_API_URL

    def generate(
        self, 
        model: str, 
        prompt: str, 
        max_tokens: int = 20, 
        temperature: float = 1.0, 
        k: int = 0, 
        p: float = 0.75,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: List[str] = None,
        return_likelihoods: str = "NONE"
    ) -> Generation:
        json_body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": k,
            "p": p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop_sequences": stop_sequences,
            "return_likelihoods": return_likelihoods
        })
        response = self.__request(json_body, cohere.GENERATE_URL, model)

        token_likelihoods = None
        if "token_likelihoods" in response:
            token_likelihoods = response["token_likelihoods"]

        return Generation(response["text"], token_likelihoods, return_likelihoods)

    def similarity(self, model: str, anchor: str, targets: List[str]) -> Similarities:
        json_body = json.dumps({
            "anchor": anchor,
            "targets": targets,
        })
        response = self.__request(json_body, cohere.SIMILARITY_URL, model)
        return Similarities(response["similarities"])

    def embed(self, model: str, texts: List[str]) -> Embeddings:
        json_body = json.dumps({
            "texts": texts,
        })
        response = self.__request(json_body, cohere.EMBED_URL, model)
        return Embeddings(response["embeddings"])

    def choose_best(self, model: str, query: str, options: List[str], mode:  str = "") -> BestChoices:
        json_body = json.dumps({
            "query": query,
            "options": options,
            "mode": mode,
        })
        response = self.__request(json_body, cohere.CHOOSE_BEST_URL, model)
        return BestChoices(response['likelihoods'], mode)

    def likelihood(self, model: str, text: List[str]) -> Likelihoods:
        json_body = json.dumps({
            "text": text,
        })
        response = self.__request(json_body, cohere.LIKELIHOOD_URL, model)
        return Likelihoods(response['likelihood'], response['token_likelihoods'])

    def __request(self, json_body, endpoint, model) -> Response:
        headers = {
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json'
        }
        url = urljoin(self.api_url, model + "/" + endpoint)
        response = requests.request("POST", url, headers=headers, data=json_body)
        res = json.loads(response.text)
        if "message" in res.keys(): # has errors
            raise CohereError(
                message=res["message"],
                http_status=response.status_code,
                headers=response.headers)
        return res
