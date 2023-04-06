import json as jsonlib
import os
import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import cohere
from cohere.error import CohereAPIError, CohereError
from cohere.logging import logger
from cohere.responses import (
    Classification,
    Classifications,
    Detokenization,
    Generations,
    StreamingGenerations,
    Tokens,
)
from cohere.responses.chat import Chat, StreamingChat
from cohere.responses.classify import Example as ClassifyExample
from cohere.responses.classify import LabelPrediction
from cohere.responses.cluster import ClusterJobResult, CreateClusterJobResponse
from cohere.responses.detectlang import DetectLanguageResponse, Language
from cohere.responses.embeddings import Embeddings
from cohere.responses.feedback import GenerateFeedbackResponse
from cohere.responses.rerank import Reranking
from cohere.responses.summarize import SummarizeResponse
from cohere.utils import is_api_key_valid


class Client:
    """Cohere Client

    Args:
        api_key (str): Your API key.
        num_workers (int): Maximal number of threads for parallelized calls.
        request_dict (dict): Additional parameters for calls with the requests library. Currently ignored in AsyncClient
        check_api_key (bool): Whether to check the api key for validity on initialization.
        client_name (str): A string to identify your application for internal analytics purposes.
        max_retries (int): maximal number of retries for requests.
        timeout (int): request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str = None,
        num_workers: int = 64,
        request_dict: dict = {},
        check_api_key: bool = True,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = cohere.COHERE_API_URL
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self._executor = ThreadPoolExecutor(num_workers)
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk"
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_version = f"v{cohere.API_VERSION}"
        if client_name:
            self.request_source += ":" + client_name

        if check_api_key:
            self.check_api_key()

    def check_api_key(self) -> Dict[str, bool]:
        """
        Checks the api key, which happens automatically during Client initialization, but not in AsyncClient.
        check_api_key raises an exception when the key is invalid, but the return value for valid keys is kept for
        backwards compatibility.
        """
        return {"valid": is_api_key_valid(self.api_key)}

    def batch_generate(self, prompts: List[str], **kwargs) -> List[Generations]:
        """A batched version of generate with multiple prompts."""
        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            res = executor.map(lambda prompt: self.generate(prompt=prompt, **kwargs), prompts)
        return list(res)

    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_vars: object = {},
        model: Optional[str] = None,
        preset: Optional[str] = None,
        num_generations: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        end_sequences: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,
        return_likelihoods: Optional[str] = None,
        truncate: Optional[str] = None,
        logit_bias: Dict[int, float] = {},
        stream: bool = False,
    ) -> Union[Generations, StreamingGenerations]:
        """Generate endpoint.
        See https://docs.cohere.ai/reference/generate for advanced arguments

        Args:
            prompt (str): Represents the prompt or text to be completed. Trailing whitespaces will be trimmed.
            model (str): (Optional) The model ID to use for generating the next reply.
            return_likelihoods (str): (Optional) One of GENERATION|ALL|NONE to specify how and if the token (log) likelihoods are returned with the response.
            preset (str): (Optional) The ID of a custom playground preset.
            num_generations (int): (Optional) The number of generations that will be returned, defaults to 1.
            max_tokens (int): (Optional) The number of tokens to predict per generation, defaults to 20.
            temperature (float): (Optional) The degree of randomness in generations from 0.0 to 5.0, lower is less random.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.\
            stream (bool): Return streaming tokens.
        Returns:
            a Generations object if stream=False, or a StreamingGenerations object if stream=True
        """
        json_body = {
            "model": model,
            "prompt": prompt,
            "prompt_vars": prompt_vars,
            "preset": preset,
            "num_generations": num_generations,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": k,
            "p": p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "end_sequences": end_sequences,
            "stop_sequences": stop_sequences,
            "return_likelihoods": return_likelihoods,
            "truncate": truncate,
            "logit_bias": logit_bias,
            "stream": stream,
        }
        response = self._request(cohere.GENERATE_URL, json=json_body, stream=stream)
        if stream:
            return StreamingGenerations(response)
        else:
            return Generations.from_dict(response=response, return_likelihoods=return_likelihoods)

    def chat(
        self,
        query: str,
        session_id: str = "",
        conversation_id: str = "",
        model: Optional[str] = None,
        return_chatlog: bool = False,
        return_prompt: bool = False,
        return_preamble: bool = False,
        chatlog_override: List[Dict[str, str]] = None,
        persona_name: str = None,
        persona_prompt: str = None,
        preamble_override: str = None,
        user_name: str = None,
        temperature: float = 0.8,
        max_tokens: int = 200,
        stream: bool = False,
    ) -> Union[Chat, StreamingChat]:
        """Returns a Chat object with the query reply.

        Args:
            query (str): The query to send to the chatbot.
            session_id (str): Deprecated, use conversation_id instead.
            conversation_id (str): (Optional) The conversation id to continue the conversation.
            model (str): (Optional) The model to use for generating the next reply.
            return_chatlog (bool): (Optional) Whether to return the chatlog.
            return_prompt (bool): (Optional) Whether to return the prompt.
            return_preamble (bool): (Optional) Whether to return the preamble.
            chatlog_override (List[Dict[str, str]]): (Optional) A list of chatlog entries to override the chatlog.
            persona_name (str): Deprecated.
            persona_prompt (str): Deprecated, use preamble_override instead.
            preamble_override (str): (Optional) A string to override the preamble.
            user_name (str): Deprecated.
            temperature (float): (Optional) The temperature to use for the next reply. The higher the temperature, the more random the reply.
            max_tokens (int): (Optional) The max tokens generated for the next reply.
            stream (bool): Return streaming tokens.
        Returns:
            a Chat object if stream=False, or a StreamingChat object if stream=True

        Examples:
            A simple chat messsage:
                >>> res = co.chat(query="Hey! How are you doing today?")
                >>> print(res.reply)
                >>> print(res.conversation_id)
            Continuing a session using a specific model:
                >>> res = co.chat(
                >>>     query="Hey! How are you doing today?",
                >>>     conversation_id="1234",
                >>>     model="command-xlarge",
                >>>     return_chatlog=True)
                >>> print(res.reply)
                >>> print(res.chatlog)
            Overriding a chat log:
                >>> res = co.chat(
                >>>     query="What about you?",
                >>>     conversation_id="1234",
                >>>     chatlog_override=[
                >>>         {'Bot': 'Hey!'},
                >>>         {'User': 'I am doing great!'},
                >>>         {'Bot': 'That is great to hear!'},
                >>>     ],
                >>>     return_chatlog=True)
                >>> print(res.reply)
                >>> print(res.chatlog)
            Streaming chat:
                >>> res = co.chat(
                >>>     query="Hey! How are you doing today?",
                >>>     stream=True)
                >>> for token in res:
                >>>     print(token)
        """
        if chatlog_override is not None:
            self._validate_chatlog_override(chatlog_override)

        if session_id != "":
            conversation_id = session_id
            logger.warning(
                "The 'session_id' parameter is deprecated and will be removed in a future version of this function. Use 'conversation_id' instead.",
            )
        if persona_prompt is not None:
            preamble_override = persona_prompt
            logger.warning(
                "The 'persona_prompt' parameter is deprecated and will be removed in a future version of this function. Use 'preamble_override' instead.",
            )
        if persona_name is not None:
            logger.warning(
                "The 'persona_name' parameter is deprecated and will be removed in a future version of this function.",
            )
        if user_name is not None:
            logger.warning(
                "The 'user_name' parameter is deprecated and will be removed in a future version of this function.",
            )

        json_body = {
            "query": query,
            "conversation_id": conversation_id,
            "model": model,
            "return_chatlog": return_chatlog,
            "return_prompt": return_prompt,
            "return_preamble": return_preamble,
            "chatlog_override": chatlog_override,
            "preamble_override": preamble_override,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        response = self._request(cohere.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return Chat.from_dict(response, query=query, client=self)

    def _validate_chatlog_override(self, chatlog_override: List[Dict[str, str]]) -> None:
        if not isinstance(chatlog_override, list):
            raise CohereError(message="chatlog_override is not a list, but it must be a list of dicts")

        for entry in chatlog_override:
            if not isinstance(entry, dict):
                raise CohereError(
                    message="chatlog_override must be a list of dicts, but it contains a non-dict element"
                )
            if len(entry) != 1:
                raise CohereError(
                    message="chatlog_override must be a list of dicts, each mapping the agent to the message."
                )

    def embed(self, texts: List[str], model: Optional[str] = None, truncate: Optional[str] = None) -> Embeddings:
        """Returns an Embeddings object for the provided texts. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            text (List[str]): A list of strings to embed.
            model (str): (Optional) The model ID to use for embedding the text.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
        """
        responses = []
        json_bodys = []

        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i : i + self.batch_size]
            json_bodys.append(
                {
                    "model": model,
                    "texts": texts_batch,
                    "truncate": truncate,
                }
            )

        meta = None
        for result in self._executor.map(lambda json_body: self._request(cohere.EMBED_URL, json=json_body), json_bodys):
            responses.extend(result["embeddings"])
            meta = result["meta"] if not meta else meta

        return Embeddings(responses, meta)

    def classify(
        self,
        inputs: List[str] = [],
        model: Optional[str] = None,
        preset: Optional[str] = None,
        examples: List[ClassifyExample] = [],
        truncate: Optional[str] = None,
    ) -> Classifications:
        """Returns a Classifications object of the inputs provided, see https://docs.cohere.ai/reference/classify for advances usage.

        Args:
            inputs (List[str]): A list of texts to classify.
            model (str): (Optional) The model ID to use for classifing the inputs.
            examples (List[ClassifyExample]): A list of ClassifyExample objects containing a text and its associated label.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
        """
        examples_dicts: list[dict[str, str]] = []
        for example in examples:
            example_dict = {"text": example.text, "label": example.label}
            examples_dicts.append(example_dict)

        json_body = {
            "model": model,
            "preset": preset,
            "inputs": inputs,
            "examples": examples_dicts,
            "truncate": truncate,
        }
        response = self._request(cohere.CLASSIFY_URL, json=json_body)

        classifications = []
        for res in response["classifications"]:
            labelObj = {}
            for label, prediction in res["labels"].items():
                labelObj[label] = LabelPrediction(prediction["confidence"])
            classifications.append(
                Classification(res["input"], res["prediction"], res["confidence"], labelObj, id=res["id"])
            )

        return Classifications(classifications, response.get("meta"))

    def summarize(
        self,
        text: str,
        model: Optional[str] = None,
        length: Optional[str] = None,
        format: Optional[str] = None,
        temperature: Optional[float] = None,
        additional_command: Optional[str] = None,
        extractiveness: Optional[str] = None,
    ) -> SummarizeResponse:
        """Returns a generated summary of the specified length for the provided text.

        Args:
            text (str): Text to summarize.
            model (str): (Optional) ID of the model.
            length (str): (Optional) One of {"short", "medium", "long"}, defaults to "medium". \
                Controls the length of the summary.
            format (str): (Optional) One of {"paragraph", "bullets"}, defaults to "paragraph". \
                Controls the format of the summary.
            extractiveness (str) One of {"high", "medium", "low"}, defaults to "high". \
                Controls how close to the original text the summary is. "High" extractiveness \
                summaries will lean towards reusing sentences verbatim, while "low" extractiveness \
                summaries will tend to paraphrase more.
            temperature (float): Ranges from 0 to 5. Controls the randomness of the output. \
                Lower values tend to generate more “predictable” output, while higher values \
                tend to generate more “creative” output. The sweet spot is typically between 0 and 1.
            additional_command (str): (Optional) Modifier for the underlying prompt, must \
                complete the sentence "Generate a summary _".

        Examples:
            Summarize a text:
                >>> res = co.summarize(text="Stock market report for today...")
                >>> print(res.summary)

            Summarize a text with a specific model and prompt:
                >>> res = co.summarize(
                >>>     text="Stock market report for today...",
                >>>     model="summarize-xlarge",
                >>>     length="long",
                >>>     format="bullets",
                >>>     temperature=0.3,
                >>>     additional_command="focusing on the highest performing stocks")
                >>> print(res.summary)
        """
        json_body = {
            "model": model,
            "text": text,
            "length": length,
            "format": format,
            "temperature": temperature,
            "additional_command": additional_command,
            "extractiveness": extractiveness,
        }
        # remove None values from the dict
        json_body = {k: v for k, v in json_body.items() if v is not None}
        response = self._request(cohere.SUMMARIZE_URL, json=json_body)

        return SummarizeResponse(id=response["id"], summary=response["summary"], meta=response["meta"])

    def batch_tokenize(self, texts: List[str]) -> List[Tokens]:
        """A batched version of tokenize"""
        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            res = executor.map(self.tokenize, texts)
        return list(res)

    def tokenize(self, text: str) -> Tokens:
        """Returns a Tokens object of the provided text, see https://docs.cohere.ai/reference/tokenize for advanced usage.

        Args:
            text (str): Text to summarize.
        """
        json_body = {"text": text}
        res = self._request(cohere.TOKENIZE_URL, json=json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"], meta=res.get("meta"))

    def batch_detokenize(self, list_of_tokens: List[List[int]]) -> List[Detokenization]:
        """A batched version of detokenize"""
        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            res = executor.map(self.detokenize, list_of_tokens)
        return list(res)

    def detokenize(self, tokens: List[int]) -> Detokenization:
        """Returns a Detokenization object of the provided tokens, see https://docs.cohere.ai/reference/detokenize for advanced usage.

        Args:
            tokens (List[int]): A list of tokens to convert to strings
        """
        json_body = {"tokens": tokens}
        res = self._request(cohere.DETOKENIZE_URL, json=json_body)
        return Detokenization(text=res["text"], meta=res.get("meta"))

    def detect_language(self, texts: List[str]) -> DetectLanguageResponse:
        """Returns a DetectLanguageResponse object of the provided texts, see https://docs.cohere.ai/reference/detect-language-1 for advanced usage.

        Args:
            texts (List[str]): A list of texts to identify language for
        """
        json_body = {
            "texts": texts,
        }
        response = self._request(cohere.DETECT_LANG_URL, json=json_body)
        results = []
        for result in response["results"]:
            results.append(Language(result["language_code"], result["language_name"]))
        return DetectLanguageResponse(results, response["meta"])

    def generate_feedback(
        self,
        request_id: str,
        good_response: bool,
        model=None,
        desired_response: str = None,
        flagged_response: bool = None,
        flagged_reason: str = None,
        prompt: str = None,
        annotator_id: str = None,
    ) -> GenerateFeedbackResponse:
        """Give feedback on a response from the Cohere Generate API to improve the model.

        Args:
            request_id (str): The request_id of the generation request to give feedback on.
            good_response (bool): Whether the response was good or not.
            model (str): (Optional) ID of the model.
            desired_response (str): (Optional) The desired response.
            flagged_response (bool): (Optional) Whether the response was flagged or not.
            flagged_reason (str): (Optional) The reason the response was flagged.
            prompt (str): (Optional) The prompt used to generate the response.
            annotator_id (str): (Optional) The ID of the annotator.

        Examples:
            A user accepts a model's suggestion in an assisted writing setting:
                >>> generations = co.generate(f"Write me a polite email responding to the one below: {email}. Response:")
                >>> if user_accepted_suggestion:
                >>>     co.generate_feedback(request_id=generations[0].id, good_response=True)

            The user edits the model's suggestion:
                >>> generations = co.generate(f"Write me a polite email responding to the one below: {email}. Response:")
                >>> if user_edits_suggestion:
                >>>     co.generate_feedback(request_id=generations[0].id, good_response=False, desired_response=user_edited_suggestion)

        """

        json_body = {
            "request_id": request_id,
            "good_response": good_response,
            "desired_response": desired_response,
            "flagged_response": flagged_response,
            "flagged_reason": flagged_reason,
            "prompt": prompt,
            "annotator_id": annotator_id,
            "model": model,
        }
        response = self._request(cohere.GENERATE_FEEDBACK_URL, json_body)
        return GenerateFeedbackResponse(id=response["id"])

    def rerank(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        model: str = None,
        top_n: Optional[int] = None,
    ) -> Reranking:
        """Returns an ordered list of documents ordered by their relevance to the provided query

        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            model (str): (Optional) The model to use for re-ranking
            top_n (int): (optional) The number of results to return, defaults to returning all results
        """
        parsed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                parsed_docs.append({"text": doc})
            elif isinstance(doc, dict) and "text" in doc:
                parsed_docs.append(doc)
            else:
                raise CohereError(
                    message='invalid format for documents, must be a list of strings or dicts with a "text" key'
                )

        json_body = {
            "query": query,
            "documents": parsed_docs,
            "model": model,
            "top_n": top_n,
            "return_documents": False,
        }

        reranking = Reranking(self._request(cohere.RERANK_URL, json=json_body))
        for rank in reranking.results:
            rank.document = parsed_docs[rank.index]
        return reranking

    def _check_response(self, json_response: Dict, headers: Dict, status_code: int):
        if "X-API-Warning" in headers:
            logger.warning(headers["X-API-Warning"])
        if "message" in json_response:  # has errors
            raise CohereAPIError(
                message=json_response["message"],
                http_status=status_code,
                headers=headers,
            )
        if 400 <= status_code < 500:
            raise CohereAPIError(
                message=f"Unexpected client error (status {status_code}): {json_response}",
                http_status=status_code,
                headers=headers,
            )
        if status_code >= 500:
            raise CohereError(message=f"Unexpected server error (status {status_code}): {json_response}")

    def _request(self, endpoint, json=None, method="POST", stream=False) -> Any:
        headers = {
            "Authorization": "BEARER {}".format(self.api_key),
            "Content-Type": "application/json",
            "Request-Source": self.request_source,
        }

        url = f"{self.api_url}/{self.api_version}/{endpoint}"
        with requests.Session() as session:
            retries = Retry(
                total=self.max_retries,
                backoff_factor=0.5,
                allowed_methods=["POST", "GET"],
                status_forcelist=cohere.RETRY_STATUS_CODES,
                raise_on_status=False,
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))
            session.mount("http://", HTTPAdapter(max_retries=retries))

            if stream:
                return session.request(method, url, headers=headers, json=json, **self.request_dict, stream=True)

            try:
                response = session.request(
                    method, url, headers=headers, json=json, timeout=self.timeout, **self.request_dict
                )
            except requests.exceptions.ConnectionError as e:
                raise CohereConnectionError(str(e)) from e
            except requests.exceptions.RequestException as e:
                raise CohereError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

            try:
                json_response = response.json()
            except jsonlib.decoder.JSONDecodeError:  # CohereAPIError will capture status
                raise CohereAPIError.from_response(response, message=f"Failed to decode json body: {response.text}")

            self._check_response(json_response, response.headers, response.status_code)
        return json_response

    def create_cluster_job(
        self,
        embeddings_url: str,
        threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
    ) -> CreateClusterJobResponse:
        """Create clustering job.

        Args:
            embeddings_url (str): File with embeddings to cluster.
            threshold (Optional[float], optional): Similarity threshold above which two texts are deemed to belong in
                the same cluster. Defaults to None.
            min_cluster_size (Optional[int], optional): Minimum number of elements in a cluster. Defaults to None.

        Returns:
            CreateClusterJobResponse: Created clustering job handler
        """

        json_body = {
            "embeddings_url": embeddings_url,
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
        }

        response = self._request(cohere.CLUSTER_JOBS_URL, json=json_body)
        return CreateClusterJobResponse.from_dict(
            response,
            wait_fn=self.wait_for_cluster_job,
        )

    def get_cluster_job(
        self,
        job_id: str,
    ) -> ClusterJobResult:
        """Get clustering job results.

        Args:
            job_id (str): Clustering job id.

        Raises:
            ValueError: "job_id" is empty

        Returns:
            ClusterJobResult: Clustering job result.
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        response = self._request(f"{cohere.CLUSTER_JOBS_URL}/{job_id}", method="GET")

        return ClusterJobResult.from_dict(response)

    def list_cluster_jobs(self) -> List[ClusterJobResult]:
        """List clustering jobs.

        Returns:
            List[ClusterJobResult]: Clustering jobs created.
        """

        response = self._request(cohere.CLUSTER_JOBS_URL, method="GET")
        return [ClusterJobResult.from_dict({"meta": response.get("meta"), **r}) for r in response["jobs"]]

    def wait_for_cluster_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> ClusterJobResult:
        """Wait for clustering job result.

        Args:
            job_id (str): Clustering job id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            ClusterJobResult: Clustering job result.
        """

        start_time = time.time()
        job = self.get_cluster_job(job_id)

        while not job.is_final_state:
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"wait_for_cluster_job timed out after {timeout} seconds")

            time.sleep(interval)
            job = self.get_cluster_job(job_id)

        return job
