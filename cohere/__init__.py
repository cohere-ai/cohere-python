import json
from typing import List
from urllib.parse import urljoin

import requests

COHERE_API_URL = "https://api.cohere.ai"
GENERATE_URL = "generate"
SIMILARITY_URL = "similarity"
EMBED_URL = "embed"
CHOOSE_BEST_URL = "choose-best"
LIKELIHOOD_URL = "likelihood"

class Generation:
    def __init__(self, text):
        self.text = text
    
    def __str__(self) -> str:
        return self.text

class Similarities:
    def __init__(self, similarities):
        self.similarities = similarities

    def __str__(self) -> str:
        return str(self.similarities)
        
class Embeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __str__(self) -> str:
        return str(self.embeddings)

class BestChoices:
    def __init__(self, likelihoods, mode):
        self.likelihoods = likelihoods
        self.mode = mode
    
    def __str__(self) -> str:
        return str(self.likelihoods)

class Likelihoods:
    def __init__(self, likelihood, token_likelihoods):
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods

    def __str__(self) -> str:
        return str(self.likelihood) + "\n" + str(self.token_likelihoods)

class CohereClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = COHERE_API_URL

    def generate(
        self, 
        model: str, 
        prompt: str, 
        max_tokens: int = 20, 
        temperature: float = 1.0, 
        k: int = 0, p: float = 0.75
    ) -> Generation:
        json_body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": k,
            "p": p,
        })
        response = self.__request(json_body, GENERATE_URL, model)
        return Generation(response["text"])

    def similarity(self, model: str, anchor: str, targets: List[str]) -> Similarities:
        json_body = json.dumps({
            "anchor": anchor,
            "targets": targets,
        })
        response = self.__request(json_body, SIMILARITY_URL, model)
        return Similarities(response["similarities"])

    def embed(self, model: str, texts: List[str]) -> Embeddings:
        json_body = json.dumps({
            "texts": texts,
        })
        response = self.__request(json_body, EMBED_URL, model)
        return Embeddings(response["embeddings"])

    def choose_best(self, model: str, query: str, options: List[str], mode:  str = "") -> BestChoices:
        json_body = json.dumps({
            "query": query,
            "options": options,
            "mode": mode,
        })
        response = self.__request(json_body, CHOOSE_BEST_URL, model)
        return BestChoices(response['likelihoods'], mode)

    def likelihood(self, model: str, text: List[str]) -> Likelihoods:
        json_body = json.dumps({
            "text": text,
        })
        response = self.__request(json_body, LIKELIHOOD_URL, model)
        return Likelihoods(response['likelihood'], response['token_likelihoods'])

    def __request(self, json_body, endpoint, model):
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

class CohereError(Exception):
    def __init__(
        self,
        message=None,
        http_status=None,
        headers=None,
    ):
        super(CohereError, self).__init__(message)

        self.message = message
        self.http_status = http_status
        self.headers = headers or {}

    def __str__(self):
        msg = self.message or "<empty message>"
        return msg

    def __repr__(self):
        return "%s(message=%r, http_status=%r, request_id=%r)" % (
            self.__class__.__name__,
            self.message,
            self.http_status,
        )

