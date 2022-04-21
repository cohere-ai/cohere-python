import json
from typing import List, Any
from urllib.parse import urljoin

import math

import requests
from requests import Response

from concurrent.futures import ThreadPoolExecutor

import cohere
from cohere.best_choices import BestChoices
from cohere.embeddings import Embeddings
from cohere.error import CohereError
from cohere.generation import Generations, Generation, TokenLikelihood
from cohere.tokenize import Tokens
from cohere.classify import Classifications, Classification, Example, Confidence

use_xhr_client = False
try:
    from js import XMLHttpRequest
    use_xhr_client = True
except ImportError:
    pass

use_go_tokenizer = False
try:
    from cohere.tokenizer import tokenizer
    use_go_tokenizer = True
except ImportError:
    pass

class Client:
    def __init__(self, api_key: str, version: str = None, num_workers: int = 8, request_dict: dict = {}) -> None:
        self.api_key = api_key
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        if version is None:
            self.cohere_version = cohere.COHERE_VERSION
        else:
            self.cohere_version = version

        try:
            res = self.check_api_key()
            if res['valid'] == False:
                raise CohereError("invalid api key")
        except CohereError as e:
            raise CohereError(
                message=e.message,
                http_status=e.http_status,
                headers=e.headers)

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
    ) -> Generations:
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

        generations: List[Generation] = []
        for gen in response['generations']:
            likelihood = None
            token_likelihoods = None
            if return_likelihoods == 'GENERATION' or return_likelihoods == 'ALL':
                likelihood = gen['likelihood']
            if 'token_likelihoods' in gen.keys():
                token_likelihoods = []
                for l in gen['token_likelihoods']:
                    token_likelihood = l['likelihood'] if 'likelihood' in l.keys() else None
                    token_likelihoods.append(TokenLikelihood(l['token'], token_likelihood))
            generations.append(Generation(gen['text'], likelihood, token_likelihoods))
        return Generations(generations, return_likelihoods)

    def embed(self, model: str, texts: List[str], truncate: str = 'NONE') -> Embeddings:
        responses = []
        json_bodys = []
        request_futures = []
        num_batch = int(math.ceil(len(texts)/self.batch_size))
        embed_url_stacked = [cohere.EMBED_URL] * num_batch
        model_stacked = [model] * num_batch

        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i:i+self.batch_size]
            json_bodys.append(json.dumps({
                'texts': texts_batch,
                'truncate': truncate,
            }))
        if use_xhr_client:
            for json_body in json_bodys:
                response = self.__request(json_body, cohere.EMBED_URL, model)
                responses.append(response['embeddings'])
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i in executor.map(self.__request, json_bodys, embed_url_stacked, model_stacked):
                    request_futures.append(i)
            for result in request_futures:
                responses.extend(result['embeddings'])

        return Embeddings(responses)

    def choose_best(self, model: str, query: str, options: List[str], mode: str = '') -> BestChoices:
        json_body = json.dumps({
            'query': query,
            'options': options,
            'mode': mode,
        })
        response = self.__request(json_body, cohere.CHOOSE_BEST_URL, model)
        return BestChoices(response['scores'], response['tokens'], response['token_log_likelihoods'], mode)

    def classify(
        self,
        model: str,
        inputs: List[str],
        examples: List[Example],
        taskDescription: str = "",
        outputIndicator: str = ""
    ) -> Classifications:
        examples_dicts: list[dict[str, str]] = []
        for example in examples:
            example_dict = {"text": example.text, "label": example.label}
            examples_dicts.append(example_dict)

        json_body = json.dumps({
            'inputs': inputs,
            'examples': examples_dicts,
            'taskDescription': taskDescription,
            'outputIndicator': outputIndicator,
        })
        response = self.__request(json_body, cohere.CLASSIFY_URL, model)

        classifications = []
        for res in response['classifications']:
            confidenceObj = []
            for i in range(len(res['confidences'])):
                confidenceObj.append(Confidence(res['confidences'][i]['option'], res['confidences'][i]['confidence']))
            Classification(res['input'], res['prediction'], confidenceObj)
            classifications.append(Classification(res['input'], res['prediction'], confidenceObj))

        return Classifications(classifications)

    def tokenize(self, model: str, text: str) -> Tokens:
        if (use_go_tokenizer):
            encoder = tokenizer.NewFromPrebuilt("coheretext-50k")
            goTokens = encoder.Encode(text)
            tokens = []
            for token in goTokens:
                tokens.append(token)
            return Tokens(tokens)
        else:
            json_body = json.dumps({
                'text': text,
            })
            response = self.__request(json_body, cohere.TOKENIZE_URL, model)
            return Tokens(response['tokens'])


    def __pyfetch(self, url, headers, json_body) -> Response:
        req = XMLHttpRequest.new()
        req.open("POST", url, False)
        for key, value in headers.items():
            req.setRequestHeader(key, value)
        try:
            req.send(json_body)
        except:
            raise CohereError(
                message=req.responseText,
                http_status=req.status,
                headers=req.getAllResponseHeaders())
        res = json.loads(req.response)
        if 'message' in res.keys():
            raise CohereError(
                message=res['message'],
                http_status=req.status,
                headers=req.getAllResponseHeaders())
        return res

    def __request(self, json_body, endpoint, model) -> Any:
        headers = {
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json',
            'Request-Source': 'python-sdk',
        }
        if self.cohere_version != '':
            headers['Cohere-Version'] = self.cohere_version

        url = urljoin(self.api_url, model + '/' + endpoint)
        if use_xhr_client:
            response = self.__pyfetch(url, headers, json_body)
            return response
        else:
            response = requests.request('POST', url, headers=headers, data=json_body, **self.request_dict)
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
