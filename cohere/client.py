import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union
from urllib.parse import urljoin

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import cohere
from cohere.chat import Chat
from cohere.classify import Classification, Classifications
from cohere.classify import Example as ClassifyExample
from cohere.classify import LabelPrediction
from cohere.detectlang import DetectLanguageResponse, Language
from cohere.detokenize import Detokenization
from cohere.embeddings import Embeddings
from cohere.error import CohereError
from cohere.feedback import Feedback
from cohere.generation import Generations
from cohere.tokenize import Tokens
from cohere.summarize import SummarizeResponse
from cohere.rerank import Reranking

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
                 num_workers: int = 64,
                 request_dict: dict = {},
                 check_api_key: bool = True,
                 client_name: str = None,
                 max_retries: int = 3) -> None:
        """
        Initialize the client.
        Args:
           * api_key (str): Your API key.
           * version (str): API version to use. Will use cohere.COHERE_VERSION by default.
           * num_workers (int): Maximal number of threads for parallelized calls.
           * request_dict (dict): Additional parameters for calls to requests.post
           * check_api_key (bool): Whether to check the api key for validity on initialization.
           * client_name (str): A string to identify your application for internal analytics purposes.
        """
        self.api_key = api_key
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self._executor = ThreadPoolExecutor(num_workers)
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = 'python-sdk'
        self.max_retries = max_retries
        if client_name:
            self.request_source += ":" + client_name

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

    def batch_generate(self, prompts: List[str], **kwargs) -> List[Generations]:
        generations: List[Generations] = []
        for prompt in prompts:
            kwargs["prompt"] = prompt
            generations.append(self.generate(**kwargs))
        return generations

    def generate(self,
                 prompt: str = None,
                 prompt_vars: object = {},
                 model: str = None,
                 preset: str = None,
                 num_generations: int = None,
                 max_tokens: int = None,
                 temperature: float = None,
                 k: int = None,
                 p: float = None,
                 frequency_penalty: float = None,
                 presence_penalty: float = None,
                 end_sequences: List[str] = None,
                 stop_sequences: List[str] = None,
                 return_likelihoods: str = None,
                 truncate: str = None,
                 logit_bias: Dict[int, float] = {}) -> Generations:
        json_body = {
            'model': model,
            'prompt': prompt,
            'prompt_vars': prompt_vars,
            'preset': preset,
            'num_generations': num_generations,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'k': k,
            'p': p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'end_sequences': end_sequences,
            'stop_sequences': stop_sequences,
            'return_likelihoods': return_likelihoods,
            'truncate': truncate,
            'logit_bias': logit_bias,
        }
        response = self._executor.submit(self.__request, cohere.GENERATE_URL, json=json_body)
        return Generations(return_likelihoods=return_likelihoods, _future=response, client=self)

    def chat(self,
             query: str,
             session_id: str = "",
             persona: str = "cohere",
             model: str = None,
             return_chatlog: bool = False) -> Chat:
        """Returns a Chat object with the query reply.

        Args:
            query (str): The query to send to the chatbot.
            session_id (str): (Optional) The session id to continue the conversation.
            persona (str): (Optional) The persona to use.
            model (str): (Optional) The model to use for generating the next reply.
            return_chatlog (bool): (Optional) Whether to return the chatlog.

        Example:
        ```
        res = co.chat(query="Hey! How are you doing today?")
        print(res.reply)
        print(res.session_id)
        ```

        Example:
        ```
        res = co.chat(
            query="Hey! How are you doing today?",
            session_id="1234",
            persona="fortune",
            model="command-xlarge",
            return_chatlog=True)
        print(res.reply)
        print(res.chatlog)
        ```
        """
        json_body = {
            'query': query,
            'session_id': session_id,
            'persona': persona,
            'model': model,
            'return_chatlog': return_chatlog,
        }
        response = self._executor.submit(self.__request, cohere.CHAT_URL, json=json_body)
        return Chat(query=query, persona=persona, _future=response, client=self, return_chatlog=return_chatlog)

    def embed(self, texts: List[str], model: str = None, truncate: str = None) -> Embeddings:
        responses = []
        json_bodys = []

        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i:i + self.batch_size]
            json_bodys.append({
                'model': model,
                'texts': texts_batch,
                'truncate': truncate,
            })
        if use_xhr_client:
            for json_body in json_bodys:
                response = self.__request(cohere.EMBED_URL, json=json_body)
                responses.append(response['embeddings'])
        else:
            for result in self._executor.map(lambda json_body: self.__request(cohere.EMBED_URL, json=json_body),
                                             json_bodys):
                responses.extend(result['embeddings'])

        return Embeddings(responses)

    def classify(self,
                 inputs: List[str] = [],
                 model: str = None,
                 preset: str = None,
                 examples: List[ClassifyExample] = [],
                 truncate: str = None) -> Classifications:
        examples_dicts: list[dict[str, str]] = []
        for example in examples:
            example_dict = {'text': example.text, 'label': example.label}
            examples_dicts.append(example_dict)

        json_body = {
            'model': model,
            'preset': preset,
            'inputs': inputs,
            'examples': examples_dicts,
            'truncate': truncate,
        }
        response = self.__request(cohere.CLASSIFY_URL, json=json_body)

        classifications = []
        for res in response['classifications']:
            labelObj = {}
            for label, prediction in res['labels'].items():
                labelObj[label] = LabelPrediction(prediction['confidence'])
            classifications.append(
                Classification(res['input'], res['prediction'], res['confidence'], labelObj, client=self, id=res["id"]))

        return Classifications(classifications)

    def summarize(self, text: str, model: str = None, length: str = None, format: str = None, temperature: float = None,
                  additional_instruction: str = None, abstractiveness: str = None) -> SummarizeResponse:
        """Return a generated summary of the specified length for the provided text.

        Args:
            text (str): Text to summarize.
            model (str): (Optional) ID of the model.
            length (str): (Optional) One of {"short", "medium", "long"}, defaults to "medium". \
                Controls the length of the summary.
            format (str): (Optional) One of {"paragraph", "bullets"}, defaults to "paragraph". \
                Controls the format of the summary.
            abstractiveness (str) One of {"high", "medium", "low"}, defaults to "high". \
                Controls how close to the original text the summary is. "Low" abstractiveness \
                summaries will lean towards reusing sentences verbatim, while "high" abstractiveness \
                summaries will tend to paraphrase more.
            temperature (float): Ranges from 0 to 5. Controls the randomness of the output. \
                Lower values tend to generate more “predictable” output, while higher values \
                tend to generate more “creative” output. The sweet spot is typically between 0 and 1.
            additional_instruction (str): (Optional) Modifier for the underlying prompt, must \
                complete the sentence "Generate a summary _".

        Example:
        ```
        res = co.summarize(text="Stock market report for today...")
        print(res.summary)
        ```

        Example:
        ```
        res = co.summarize(
            text="Stock market report for today...",
            model="summarize-xlarge",
            length="long",
            format="bullets",
            temperature=0.9,
            additional_instruction="focusing on the highest performing stocks")
        print(res.summary)
        ```
        """
        json_body = {
            'model': model,
            'text': text,
            'length': length,
            'format': format,
            'temperature': temperature,
            'additional_instruction': additional_instruction,
            'abstractiveness': abstractiveness,
        }
        # remove None values from the dict
        json_body = {k: v for k, v in json_body.items() if v is not None}
        response = self.__request(cohere.SUMMARIZE_URL, json=json_body)

        return SummarizeResponse(id=response["id"], summary=response["summary"])

    def batch_tokenize(self, texts: List[str]) -> List[Tokens]:
        return [self.tokenize(t) for t in texts]

    def tokenize(self, text: str) -> Tokens:
        json_body = {'text': text}
        return Tokens(_future=self._executor.submit(self.__request, cohere.TOKENIZE_URL, json=json_body))

    def batch_detokenize(self, list_of_tokens: List[List[int]]) -> List[Detokenization]:
        return [self.detokenize(t) for t in list_of_tokens]

    def detokenize(self, tokens: List[int]) -> Detokenization:
        json_body = {'tokens': tokens}
        return Detokenization(_future=self._executor.submit(self.__request, cohere.DETOKENIZE_URL, json=json_body))

    def detect_language(self, texts: List[str]) -> List[Language]:
        json_body = {
            "texts": texts,
        }
        response = self.__request(cohere.DETECT_LANG_URL, json=json_body)
        results = []
        for result in response["results"]:
            results.append(Language(result["language_code"], result["language_name"]))
        return DetectLanguageResponse(results)

    def feedback(self, id: str, good_response: bool, desired_response: str = "", feedback: str = "") -> Feedback:
        """Give feedback on a response from the Cohere API to improve the model.

        Can be used programmatically like so:

        Example: a user accepts a model's suggestion in an assisted writing setting
        ```
        generations = co.generate(f"Write me a polite email responding to the one below:\n{email}\n\nResponse:")
        if user_accepted_suggestion:
            generations[0].feedback(good_response=True)
        ```

        Example: the user edits the model's suggestion
        ```
        generations = co.generate(f"Write me a polite email responding to the one below:\n{email}\n\nResponse:")
        if user_edits_suggestion:
            generations[0].feedback(good_response=False, desired_response=user_edited_response)
        ```

        Args:
            id (str): the `id` associated with a generation from the Cohere API
            good_response (bool): a boolean indicator as to whether the generation was good (True) or bad (False).
            desired_response (str): an optional string of the response expected. To be used when a mistake has been
            made or a better response exists.
            feedback (str): an optional natural language description of the specific feedback about this generation.

        Returns:
            Feedback: a Feedback object
        """
        json_body = {
            'id': id,
            'good_response': good_response,
            'desired_response': desired_response,
            'feedback': feedback,
        }
        self.__request(cohere.FEEDBACK_URL, json_body)
        return Feedback(id=id, good_response=good_response, desired_response=desired_response, feedback=feedback)

    def rerank(self,
               query: str,
               documents: Union[List[str], List[Dict[str, Any]]],
               top_n: int = None) -> Reranking:
        """Returns an ordered list of documents ordered by their relevance to the provided query

        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            top_n (int): (optional) The number of results to return, defaults to returning all results
        """
        parsed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                parsed_docs.append({'text': doc})
            elif isinstance(doc, dict) and 'text' in doc:
                parsed_docs.append(doc)
            else:
                raise CohereError(
                    message='invalid format for documents, must be a list of strings or dicts with a "text" key')

        json_body = {
            "query": query,
            "documents": parsed_docs,
            "top_n": top_n,
            "return_documents": False
        }
        reranking = Reranking(self.__request(cohere.RERANK_URL, json=json_body))
        for rank in reranking.results:
            rank.document = parsed_docs[rank.index]
        return reranking

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

    def __request(self, endpoint, json=None) -> Any:
        headers = {
            'Authorization': 'BEARER {}'.format(self.api_key),
            'Content-Type': 'application/json',
            'Request-Source': self.request_source,
        }
        if self.cohere_version != '':
            headers['Cohere-Version'] = self.cohere_version

        url = urljoin(self.api_url, endpoint)
        if use_xhr_client:
            response = self.__pyfetch(url, headers, json.dumps(json))
            self.__print_warning_msg(response)
            return response
        else:
            with requests.Session() as session:
                retries = Retry(
                    total=self.max_retries,
                    backoff_factor=0.5,
                    allowed_methods=['POST', 'GET'],
                    status_forcelist=[429, 500, 502, 503, 504]
                )
                session.mount('https://', HTTPAdapter(max_retries=retries))
                session.mount('http://', HTTPAdapter(max_retries=retries))

                response = session.request('POST', url, headers=headers, json=json, **self.request_dict)
                try:
                    res = response.json()
                except Exception:
                    raise CohereError(
                        message=response.text, http_status=response.status_code, headers=response.headers)
                if 'message' in res:  # has errors
                    raise CohereError(
                        message=res['message'], http_status=response.status_code, headers=response.headers)
                self.__print_warning_msg(response)

        return res
