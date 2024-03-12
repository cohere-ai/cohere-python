import json as jsonlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Union

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import cohere
from cohere.custom_model_dataset import CustomModelDataset
from cohere.error import CohereAPIError, CohereConnectionError, CohereError
from cohere.logging import logger
from cohere.responses import (
    Classification,
    Classifications,
    Codebook,
    Detokenization,
    Generations,
    LogLikelihoods,
    StreamingGenerations,
    Tokens,
)
from cohere.responses.chat import Chat, ChatRequestToolResultsItem, StreamingChat, Tool
from cohere.responses.classify import Example as ClassifyExample
from cohere.responses.classify import LabelPrediction
from cohere.responses.cluster import ClusterJobResult
from cohere.responses.connector import Connector
from cohere.responses.custom_model import (
    CUSTOM_MODEL_INTERNAL_STATUS_MAPPING,
    CUSTOM_MODEL_PRODUCT_MAPPING,
    CUSTOM_MODEL_STATUS,
    CUSTOM_MODEL_TYPE,
    INTERNAL_CUSTOM_MODEL_TYPE,
    CustomModel,
    HyperParametersInput,
    ModelMetric,
)
from cohere.responses.dataset import Dataset, DatasetUsage, ParseInfo
from cohere.responses.detectlang import DetectLanguageResponse
from cohere.responses.embed_job import EmbedJob
from cohere.responses.embeddings import EmbeddingResponses, Embeddings
from cohere.responses.feedback import (
    GenerateFeedbackResponse,
    GeneratePreferenceFeedbackResponse,
    PreferenceRating,
)
from cohere.responses.rerank import Reranking
from cohere.responses.summarize import SummarizeResponse
from cohere.utils import is_api_key_valid, threadpool_map, wait_for_job


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
        api_url (str): override the default api url from the default cohere.COHERE_API_URL
    """

    def __init__(
        self,
        api_key: str = None,
        num_workers: int = 64,
        request_dict: dict = {},
        check_api_key: bool = True,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 300,
        api_url: str = None,
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = api_url or os.getenv("CO_API_URL", cohere.COHERE_API_URL)
        self.batch_size = cohere.COHERE_EMBED_BATCH_SIZE
        self._executor = ThreadPoolExecutor(num_workers)
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk-" + cohere.SDK_VERSION
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

    def loglikelihood(
        self,
        prompt: Optional[str] = None,
        completion: Optional[str] = None,
        model: Optional[str] = None,
    ) -> LogLikelihoods:
        """Calculates the token log-likelihood for a provided prompt and completion.
        Using this endpoint instead of co.generate with max_tokens=0 will guarantee that any required tokens such as <EOP_TOKEN>
        are correctly inserted, and makes it easier to retrieve only the completion log-likelihood.

        Args:
            prompt (str): The prompt
            completion (str): (Optional) The completion
            model (str): (Optional) The model to use for calculating the log-likelihoods
        """
        json_body = {"model": model, "prompt": prompt, "completion": completion}
        response = self._request(cohere.LOGLIKELIHOOD_URL, json=json_body)
        return LogLikelihoods(response["prompt_tokens"], response["completion_tokens"])

    def batch_generate(
        self, prompts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[Generations, Exception]]:
        """A batched version of generate with multiple prompts.

        Args:
            prompts: list of prompts
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `generate`
        """
        return threadpool_map(
            self.generate,
            [dict(prompt=prompt, **kwargs) for prompt in prompts],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

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
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
            stream (bool): Return streaming tokens.
        Returns:
            if stream=False: a Generations object
            if stream=True: a StreamingGenerations object including:
                id (str): The id of the whole generation call
                generations (Generations): same as the response when stream=False
                finish_reason (string) possible values:
                    COMPLETE: when the stream successfully completed
                    ERROR: when an error occurred during streaming
                    ERROR_TOXIC: when the stream was halted due to toxic output.
                    ERROR_LIMIT: when the context is too big to generate.
                    USER_CANCEL: when the user has closed the stream / cancelled the request
                    MAX_TOKENS: when the max tokens limit was reached.
                texts (List[str]): list of segments of text streamed back from the API

        Examples:
            A simple generate message:
                >>> res = co.generate(prompt="Hey! How are you doing today?")
                >>> print(res.text)
            Streaming generate:
                >>> res = co.generate(
                >>>     prompt="Hey! How are you doing today?",
                >>>     stream=True)
                >>> for token in res:
                >>>     print(token)
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
            "stream": stream,
        }
        response = self._request(cohere.GENERATE_URL, json=json_body, stream=stream)
        if stream:
            return StreamingGenerations(response)
        else:
            return Generations.from_dict(response=response, return_likelihoods=return_likelihoods)

    def chat(
        self,
        message: Optional[str] = None,
        conversation_id: Optional[str] = "",
        model: Optional[str] = None,
        return_chat_history: Optional[bool] = False,
        return_prompt: Optional[bool] = False,
        return_preamble: Optional[bool] = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
        preamble: Optional[str] = None,
        user_name: Optional[str] = None,
        temperature: Optional[float] = 0.8,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        p: Optional[float] = None,
        k: Optional[float] = None,
        search_queries_only: Optional[bool] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        citation_quality: Optional[str] = None,
        prompt_truncation: Optional[str] = None,
        connectors: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Tool]] = None,
        tool_results: Optional[List[ChatRequestToolResultsItem]] = None,
        raw_prompting: Optional[bool] = False,
    ) -> Union[Chat, StreamingChat]:
        """Returns a Chat object with the query reply.

        Args:
            message (str): The message to send to the chatbot.

            stream (bool): Return streaming tokens.
            conversation_id (str): (Optional) To store a conversation then create a conversation id and use it for every related request.

            preamble (str): (Optional) A string to override the preamble.
            chat_history (List[Dict[str, str]]): (Optional) A list of entries used to construct the conversation. If provided, these messages will be used to build the prompt and the conversation_id will be ignored so no data will be stored to maintain state.

            model (str): (Optional) The model to use for generating the response.
            temperature (float): (Optional) The temperature to use for the response. The higher the temperature, the more random the response.
            p (float): (Optional) The nucleus sampling probability.
            k (float): (Optional) The top-k sampling probability.
            max_tokens (int): (Optional) The max tokens generated for the next reply.

            return_chat_history (bool): (Optional) Whether to return the chat history.
            return_prompt (bool): (Optional) Whether to return the prompt.
            return_preamble (bool): (Optional) Whether to return the preamble.

            search_queries_only (bool) : (Optional) When true, the response will only contain a list of generated search queries, but no search will take place, and no reply from the model to the user's message will be generated.
            documents (List[Dict[str, str]]): (Optional) Documents to use to generate grounded response with citations. Example:
                documents=[
                    {
                        "id": "national_geographic_everest",
                        "title": "Height of Mount Everest",
                        "snippet": "The height of Mount Everest is 29,035 feet",
                        "url": "https://education.nationalgeographic.org/resource/mount-everest/",
                    },
                    {
                        "id": "national_geographic_mariana",
                        "title": "Depth of the Mariana Trench",
                        "snippet": "The depth of the Mariana Trench is 36,070 feet",
                        "url": "https://www.nationalgeographic.org/activity/mariana-trench-deepest-place-earth",
                    },
                ],
            connectors (List[Dict[str, str]]): (Optional) When specified, the model's reply will be enriched with information found by quering each of the connectors (RAG). Example: connectors=[{"id": "web-search"}]
            citation_quality (str): (Optional) Dictates the approach taken to generating citations by allowing the user to specify whether they want "accurate" results or "fast" results. Defaults to "accurate".
            prompt_truncation (str): (Optional) Dictates how the prompt will be constructed. With `prompt_truncation` set to "AUTO", some elements from `chat_history` and `documents` will be dropped in attempt to construct a prompt that fits within the model's context length limit. With `prompt_truncation` set to "OFF", no elements will be dropped. If the sum of the inputs exceeds the model's context length limit, a `TooManyTokens` error will be returned.
        Returns:
            a Chat object if stream=False, or a StreamingChat object if stream=True

        Examples:
            A simple chat message:
                >>> res = co.chat(message="Hey! How are you doing today?")
                >>> print(res.text)
            Continuing a session using a specific model:
                >>> res = co.chat(
                >>>     message="Hey! How are you doing today?",
                >>>     conversation_id="1234",
                >>>     model="command",
                >>>     return_chat_history=True)
                >>> print(res.text)
                >>> print(res.chat_history)
            Streaming chat:
                >>> res = co.chat(
                >>>     message="Hey! How are you doing today?",
                >>>     stream=True)
                >>> for token in res:
                >>>     print(token)
            Stateless chat with chat history:
                >>> res = co.chat(
                >>>     message="Tell me a joke!",
                >>>     chat_history=[
                >>>         {'role': 'User', message': 'Hey! How are you doing today?'},
                >>>         {'role': 'Chatbot', message': 'I am doing great! How can I help you?'},
                >>>     ],
                >>>     return_prompt=True)
                >>> print(res.text)
                >>> print(res.prompt)
            Chat message with documents to use to generate the response:
                >>> res = co.chat(
                >>>     "How deep in the Mariana Trench",
                >>>     documents=[
                >>>         {
                >>>            "id": "national_geographic_everest",
                >>>            "title": "Height of Mount Everest",
                >>>            "snippet": "The height of Mount Everest is 29,035 feet",
                >>>            "url": "https://education.nationalgeographic.org/resource/mount-everest/",
                >>>         },
                >>>         {
                >>>             "id": "national_geographic_mariana",
                >>>             "title": "Depth of the Mariana Trench",
                >>>             "snippet": "The depth of the Mariana Trench is 36,070 feet",
                >>>             "url": "https://www.nationalgeographic.org/activity/mariana-trench-deepest-place-earth",
                >>>         },
                >>>       ])
                >>> print(res.text)
                >>> print(res.citations)
                >>> print(res.documents)
            Chat message with connector to query and use the results to generate the response:
                >>> res = co.chat(
                >>>     "What is the height of Mount Everest?",
                >>>      connectors=[{"id": "web-search"})
                >>> print(res.text)
                >>> print(res.citations)
                >>> print(res.documents)
            Generate search queries for fetching documents to use in chat:
                >>> res = co.chat(
                >>>     "What is the height of Mount Everest?",
                >>>      search_queries_only=True)
                >>> if res.is_search_required:
                >>>      print(res.search_queries)
        """

        json_body = {
            "message": message,
            "conversation_id": conversation_id,
            "model": model,
            "return_chat_history": return_chat_history,
            "return_prompt": return_prompt,
            "return_preamble": return_preamble,
            "chat_history": chat_history,
            "preamble": preamble,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "user_name": user_name,
            "p": p,
            "k": k,
            "search_queries_only": search_queries_only,
            "documents": documents,
            "connectors": connectors,
            "tools": tools,
            "tool_results": tool_results,
            "raw_prompting": raw_prompting,
        }
        if citation_quality is not None:
            json_body["citation_quality"] = citation_quality
        if prompt_truncation is not None:
            json_body["prompt_truncation"] = prompt_truncation

        response = self._request(cohere.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return Chat.from_dict(response, message=message, client=self)

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        truncate: Optional[str] = None,
        input_type: Optional[str] = None,
        embedding_types: Optional[List[str]] = None,
    ) -> Embeddings:
        """Returns an Embeddings object for the provided texts. Visit https://cohere.ai/embed to learn about embeddings.

        Args:
            text (List[str]): A list of strings to embed.
            model (str): (Optional) The model ID to use for embedding the text.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
            input_type (str): (Optional) One of "classification", "clustering", "search_document", "search_query". The type of input text provided to embed.
            embedding_types (List[str]): (Optional) Specifies the types of embeddings you want to get back. Not required and default is None, which returns the float embeddings in the response's embeddings field. Can be one or more of the following types: "float", "int8", "uint8", "binary", "ubinary".
        """
        embedding_responses = EmbeddingResponses()
        json_bodys = []

        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i : i + self.batch_size]
            json_bodys.append(
                {
                    "model": model,
                    "texts": texts_batch,
                    "truncate": truncate,
                    "input_type": input_type,
                    "embedding_types": embedding_types,
                }
            )

        meta = None
        for result in self._executor.map(lambda json_body: self._request(cohere.EMBED_URL, json=json_body), json_bodys):
            embedding_responses.add_response(result)
            meta = result["meta"] if not meta else meta

        return Embeddings(
            embeddings=embedding_responses.get_embeddings(),
            response_type=embedding_responses.response_type,
            meta=meta,
        )

    def codebook(
        self,
        model: Optional[str] = None,
        compression_codebook: Optional[str] = "default",
    ) -> Codebook:
        """Returns a codebook object for the provided model. Visit https://cohere.ai/embed to learn about compressed embeddings and codebooks.

        Args:
            model (str): (Optional) The model ID to use for embedding the text.
            compression_codebook (str): (Optional) The compression codebook to use for compressed embeddings. Defaults to "default".
        """
        json_body = {
            "model": model,
            "compression_codebook": compression_codebook,
        }
        response = self._request(cohere.CODEBOOK_URL, json=json_body)
        return Codebook(response["codebook"], response["meta"])

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
        examples_dicts = [{"text": example.text, "label": example.label} for example in examples]

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
                Classification(
                    input=res["input"],
                    predictions=res.get("predictions", None),
                    confidences=res.get("confidences", None),
                    prediction=res.get("prediction", None),
                    confidence=res.get("confidence", None),
                    labels=labelObj,
                    classification_type=res.get("classification_type", "single-label"),
                    id=res["id"],
                )
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

    def batch_tokenize(self, texts: List[str], return_exceptions=False, **kwargs) -> List[Union[Tokens, Exception]]:
        """A batched version of tokenize.

        Args:
            texts: list of texts
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `tokenize`
        """
        return threadpool_map(
            self.tokenize,
            [dict(text=text, **kwargs) for text in texts],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

    def tokenize(self, text: str, model: Optional[str] = None) -> Tokens:
        """Returns a Tokens object of the provided text, see https://docs.cohere.ai/reference/tokenize for advanced usage.

        Args:
            text (str): Text to summarize.
            model (str): An optional model name that will ensure that the tokenization uses the tokenizer used by that model, which can be critical for counting tokens properly.
        """
        json_body = {"text": text, "model": model}
        res = self._request(cohere.TOKENIZE_URL, json=json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"], meta=res.get("meta"))

    def batch_detokenize(
        self, list_of_tokens: List[List[int]], return_exceptions=False, **kwargs
    ) -> List[Union[Detokenization, Exception]]:
        """A batched version of detokenize.

        Args:
            list_of_tokens: list of list of tokens
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `detokenize`
        """
        return threadpool_map(
            self.detokenize,
            [dict(tokens=tokens, **kwargs) for tokens in list_of_tokens],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

    def detokenize(self, tokens: List[int], model: Optional[str] = None) -> Detokenization:
        """Returns a Detokenization object of the provided tokens, see https://docs.cohere.ai/reference/detokenize for advanced usage.

        Args:
            tokens (List[int]): A list of tokens to convert to strings
            model (str): An optional model name. This will ensure that the detokenization is done by the tokenizer used by that model.
        """
        json_body = {"tokens": tokens, "model": model}
        res = self._request(cohere.DETOKENIZE_URL, json=json_body)
        return Detokenization(text=res["text"], meta=res.get("meta"))

    def detect_language(self, texts: List[str]) -> DetectLanguageResponse:
        """
        This API is deprecated.
        """
        raise DeprecationWarning("The detect_language API is no longer supported")

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

    def generate_preference_feedback(
        self,
        ratings: List[PreferenceRating],
        model=None,
        prompt: str = None,
        annotator_id: str = None,
    ) -> GeneratePreferenceFeedbackResponse:
        """Give preference feedback on a response from the Cohere Generate API to improve the model.

        Args:
            ratings (List[PreferenceRating]): A list of PreferenceRating objects.
            model (str): (Optional) ID of the model.
            prompt (str): (Optional) The prompt used to generate the response.
            annotator_id (str): (Optional) The ID of the annotator.

        Examples:
            A user accepts a model's suggestion in an assisted writing setting, and prefers it to a second suggestion:
            >>> generations = co.generate(f"Write me a polite email responding to the one below: {email}. Response:", num_generations=2)
            >>> if user_accepted_idx: // prompt user for which generation they prefer
            >>>    ratings = []
            >>>    if user_accepted_idx == 0:
            >>>        ratings.append(PreferenceRating(request_id=0, rating=1))
            >>>        ratings.append(PreferenceRating(request_id=1, rating=0))
            >>>    else:
            >>>        ratings.append(PreferenceRating(request_id=0, rating=0))
            >>>        ratings.append(PreferenceRating(request_id=1, rating=1))
            >>>    co.generate_preference_feedback(ratings=ratings)
        """
        ratings_dicts = []
        for rating in ratings:
            ratings_dicts.append(asdict(rating))

        json_body = {
            "ratings": ratings_dicts,
            "prompt": prompt,
            "annotator_id": annotator_id,
            "model": model,
        }
        response = self._request(cohere.GENERATE_PREFERENCE_FEEDBACK_URL, json_body)
        return GenerateFeedbackResponse(id=response["id"])

    def rerank(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        model: str,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
    ) -> Reranking:
        """Returns an ordered list of documents ordered by their relevance to the provided query

        Args:
            query (str): The search query
            documents (list[str], list[dict]): The documents to rerank
            model (str): The model to use for re-ranking
            top_n (int): (optional) The number of results to return, defaults to returning all results
            max_chunks_per_doc (int): (optional) The maximum number of chunks derived from a document
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
            "max_chunks_per_doc": max_chunks_per_doc,
        }

        reranking = Reranking(self._request(cohere.RERANK_URL, json=json_body))
        for rank in reranking.results:
            rank.document = parsed_docs[rank.index]
        return reranking

    def create_dataset(
        self,
        name: str,
        data: BinaryIO,
        dataset_type: str,
        eval_data: Optional[BinaryIO] = None,
        keep_fields: Union[str, List[str]] = None,
        optional_fields: Union[str, List[str]] = None,
        parse_info: Optional[ParseInfo] = None,
    ) -> Dataset:
        """Returns a Dataset given input data

        Args:
            name (str): The name of your dataset
            data (BinaryIO): The data to be uploaded and validated
            dataset_type (str): The type of dataset you want to upload
            eval_data (BinaryIO): (optional) If the dataset type supports it upload evaluation data
            keep_fields (Union[str, List[str]]): (optional) A list of fields you want to keep in the dataset that are required
            optional_fields (Union[str, List[str]]): (optional) A list of fields you want to keep in the dataset that are optional
            parse_info: ParseInfo: (optional) information on how to parse the raw data
        Returns:
            Dataset: Dataset object.
        """
        files = {"data": data}
        if eval_data:
            files["eval_data"] = eval_data
        params = {
            "name": name,
            "type": dataset_type,
            "keep_fields": keep_fields,
            "optional_fields": optional_fields,
        }
        if parse_info:
            params.update(parse_info.get_params())

        logger.warning("uploading file, starting validation...")
        create_response = self._request(cohere.DATASET_URL, files=files, params=params)
        logger.warning(f"{create_response['id']} was uploaded")
        return self.get_dataset(id=create_response["id"])

    def get_dataset(self, id: str) -> Dataset:
        """Returns a Dataset given a dataset id

        Args:
            id (str): The name of id of your dataset

        Returns:
            Dataset: Dataset object.
        """
        if not id:
            raise CohereError(message="id must not be empty")
        response = self._request(f"{cohere.DATASET_URL}/{id}", method="GET")
        return Dataset.from_dict(response["dataset"], wait_fn=self.wait_for_dataset)

    def list_datasets(self, dataset_type: str = None, limit: int = None, offset: int = None) -> List[Dataset]:
        """Returns a list of your Datasets

        Args:
            dataset_type (str): (optional) The dataset_type to filter on
            limit (int): (optional) The max number of datasets to return
            offset (int): (optional) The number of datasets to offset by

        Returns:
            List[Dataset]: List of Dataset objects.
        """
        param_dict = {
            "dataset_type": dataset_type,
            "limit": limit,
            "offset": offset,
        }
        response = self._request(f"{cohere.DATASET_URL}", method="GET", params=param_dict)
        return [
            Dataset.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_dataset)
            for r in (response.get("datasets") or [])
        ]

    def delete_dataset(self, id: str) -> None:
        """Deletes your dataset

        Args:
            id (str): The id of the dataset to delete
        """
        self._request(f"{cohere.DATASET_URL}/{id}", method="DELETE")

    def get_dataset_usage(self) -> DatasetUsage:
        """Gets your total storage used in datasets

        Returns:
            DatasetUsage: Object containg current dataset usage
        """
        response = self._request(f"{cohere.DATASET_URL}/usage", method="GET")
        return DatasetUsage.from_dict(response)

    def wait_for_dataset(
        self,
        dataset_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> Dataset:
        """Wait for Dataset validation result.

        Args:
            dataset_id (str): Dataset id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            Dataset: Dataset object.
        """

        return wait_for_job(
            get_job=partial(self.get_dataset, dataset_id),
            timeout=timeout,
            interval=interval,
        )

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

    def _request(self, endpoint, json=None, files=None, method="POST", stream=False, params=None) -> Any:
        headers = {
            "Authorization": "BEARER {}".format(self.api_key),
            "Request-Source": self.request_source,
        }
        if json:
            headers["Content-Type"] = "application/json"

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
                    method,
                    url,
                    headers=headers,
                    json=json,
                    files=files,
                    timeout=self.timeout,
                    params=params,
                    **self.request_dict,
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
        input_dataset_id: str = None,
        embeddings_url: str = None,
        min_cluster_size: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        is_deterministic: Optional[bool] = None,
        generate_descriptions: Optional[bool] = None,
    ) -> ClusterJobResult:
        """Create clustering job.

        Args:
            input_dataset_id (str): Id of the dataset to cluster.
            embeddings_url (str): File with embeddings to cluster.
            min_cluster_size (Optional[int], optional): Minimum number of elements in a cluster. Defaults to 10.
            n_neighbors (Optional[int], optional): Number of nearest neighbors used by UMAP to establish the
                local structure of the data. Defaults to 15. For more information, please refer to
                https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
            is_deterministic (Optional[bool], optional): Determines whether the output of the cluster job is
                deterministic. Defaults to True.
            generate_descriptions (Optional[bool], optional): Determines whether to generate cluster descriptions. Defaults to False.

        Returns:
            ClusterJobResult: Created clustering job
        """

        json_body = {
            "input_dataset_id": input_dataset_id,
            "embeddings_url": embeddings_url,
            "min_cluster_size": min_cluster_size,
            "n_neighbors": n_neighbors,
            "is_deterministic": is_deterministic,
            "generate_descriptions": generate_descriptions,
        }

        response = self._request(cohere.CLUSTER_JOBS_URL, json=json_body)
        cluster_job = self.get_cluster_job(response.get("job_id"))
        return cluster_job

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

        return ClusterJobResult.from_dict(response, wait_fn=self.wait_for_cluster_job)

    def list_cluster_jobs(self) -> List[ClusterJobResult]:
        """List clustering jobs.

        Returns:
            List[ClusterJobResult]: Clustering jobs created.
        """

        response = self._request(cohere.CLUSTER_JOBS_URL, method="GET")
        return [
            ClusterJobResult.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_cluster_job)
            for r in response["jobs"]
        ]

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

        return wait_for_job(
            get_job=partial(self.get_cluster_job, job_id),
            timeout=timeout,
            interval=interval,
        )

    def create_embed_job(
        self,
        dataset_id: str,
        model: str,
        input_type: str,
        name: Optional[str] = None,
        truncate: Optional[str] = None,
        embedding_types: Optional[List[str]] = None,
    ) -> EmbedJob:
        """Create embed job.

        Args:
            dataset_id (str): ID of the dataset to embed.
            model (str): ID of the model to use for embedding the text.
            input_type (str): One of "classification", "clustering", "search_document", "search_query". The type of input text provided to embed.
            name (Optional[str], optional): The name of the embed job. Defaults to None.
            truncate (Optional[str], optional): How the API handles text longer than the maximum token length. Defaults to None.
            embedding_types (List[str]): (Optional) Specifies the types of embeddings you want to get back. Not required and default is None, which returns the float embeddings in the response's embeddings field. Can be one or more of the following types: "float", "int8", "uint8", "binary", "ubinary".

        Returns:
            EmbedJob: The created embed job
        """

        json_body = {
            "dataset_id": dataset_id,
            "name": name,
            "model": model,
            "truncate": truncate,
            "input_type": input_type,
            "embedding_types": embedding_types,
        }

        response = self._request(cohere.EMBED_JOBS_URL, json=json_body)
        embed_job = self.get_embed_job(response.get("job_id"))

        return embed_job

    def list_embed_jobs(self) -> List[EmbedJob]:
        """List embed jobs.

        Returns:
            List[EmbedJob]: Embed jobs.
        """

        response = self._request(f"{cohere.EMBED_JOBS_URL}", method="GET")
        return [
            EmbedJob.from_dict({"meta": response.get("meta"), **r}, wait_fn=self.wait_for_embed_job)
            for r in response["embed_jobs"]
        ]

    def get_embed_job(self, job_id: str) -> EmbedJob:
        """Get embed job.

        Args:
            job_id (str): Embed job id.

        Raises:
            ValueError: "job_id" is empty

        Returns:
            EmbedJob: Embed job.
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        response = self._request(f"{cohere.EMBED_JOBS_URL}/{job_id}", method="GET")
        job = EmbedJob.from_dict(response, wait_fn=self.wait_for_embed_job)
        if response.get("output_dataset_id"):
            job.output = self.get_dataset(response.get("output_dataset_id"))
        return job

    def cancel_embed_job(self, job_id: str) -> None:
        """Cancel embed job.

        Args:
            job_id (str): Embed job id.

        Raises:
            ValueError: "job_id" is empty
        """

        if not job_id.strip():
            raise ValueError('"job_id" is empty')

        self._request(f"{cohere.EMBED_JOBS_URL}/{job_id}/cancel", method="POST", json={})

    def wait_for_embed_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        interval: float = 10,
    ) -> EmbedJob:
        """Wait for embed job completion.

        Args:
            job_id (str): Embed job id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            EmbedJob: Embed job.
        """

        return wait_for_job(
            get_job=partial(self.get_embed_job, job_id),
            timeout=timeout,
            interval=interval,
        )

    def create_custom_model(
        self,
        name: str,
        model_type: CUSTOM_MODEL_TYPE,
        dataset: Union[Dataset, str],
        base_model: Optional[str] = None,
        hyperparameters: Optional[HyperParametersInput] = None,
    ) -> CustomModel:
        """Create a new custom model

        Args:
            name (str): name of your custom model, has to be unique across your organization
            model_type (GENERATIVE, CLASSIFY, RERANK): type of custom model
            dataset (Dataset, str): A dataset or dataset id for your training.
            base_model (str): base model to use for your custom model.
                For generative and classify models, `base_model` has to be None (no option available for now)
                For rerank models, you can choose between `english` and `multilingual`. Defaults to `english` if not specified.
                    The English model is better for English, while the multilingual model should be picked if a non-negligible part of queries/documents
                    will be in other languages
            hyperparameters (HyperParametersInput): adjust hyperparameters for your custom model. Only for generative custom models.
        Returns:
            CustomModel: the custom model that was created

        Examples:
             prompt completion custom model with dataset
                >>> co = cohere.Client("YOUR_API_KEY")
                >>> ds = co.create_dataset(name="prompt-completion-datset", data=open("/path/to/your/file.csv", "rb"), dataset_type="prompt-completion-finetune-input")
                >>> ds.await_validation()
                >>> co.create_custom_model("prompt-completion-ft", model_type="GENERATIVE", train_dataset=ds.id)

             classification custom model with train and evaluation data
                >>> co = cohere.Client("YOUR_API_KEY")
                >>> ds = co.create_dataset(name="classify-datset", data=open("train_file.csv", "rb"), eval_data=open("eval_file", "rb"), dataset_type="single-label-classification-finetune-input")
                >>> ds.await_validation()
                >>> co.create_custom_model("classify-ft", model_type="CLASSIFY", train_dataset=ds.id)
        """
        internal_custom_model_type = CUSTOM_MODEL_PRODUCT_MAPPING[model_type]

        json = {
            "name": name,
            "settings": {
                "trainFiles": [],
                "evalFiles": [],
                "baseModel": base_model,
                "finetuneType": internal_custom_model_type,
            },
        }
        if hyperparameters:
            json["settings"]["hyperparameters"] = {
                "earlyStoppingPatience": hyperparameters.get("early_stopping_patience"),
                "earlyStoppingThreshold": hyperparameters.get("early_stopping_threshold"),
                "trainBatchSize": hyperparameters.get("train_batch_size"),
                "trainEpochs": hyperparameters.get("train_epochs"),
                "learningRate": hyperparameters.get("learning_rate"),
            }

        if isinstance(dataset, Dataset):
            if not dataset.has_terminal_status():
                dataset.wait()
            json["settings"]["datasetID"] = dataset.id
        elif isinstance(dataset, str):
            dataset = self.get_dataset(dataset)
            if not dataset.has_terminal_status():
                dataset.wait()
            json["settings"]["datasetID"] = dataset.id
        elif isinstance(dataset, CustomModelDataset):
            logger.warning("`CustomModelDataset` is deprecated, use `Dataset` instead.")
            remote_path = self._upload_dataset(
                dataset.get_train_data(), name, dataset.train_file_name(), internal_custom_model_type
            )
            json["settings"]["trainFiles"].append({"path": remote_path, **dataset.file_config()})
            if dataset.has_eval_file():
                remote_path = self._upload_dataset(
                    dataset.get_eval_data(), name, dataset.eval_file_name(), internal_custom_model_type
                )
                json["settings"]["evalFiles"].append({"path": remote_path, **dataset.file_config()})
        else:
            raise CohereError(f"unsupported type for dataset {type(dataset)}")

        response = self._request(f"{cohere.CUSTOM_MODEL_URL}/CreateFinetune", method="POST", json=json)
        return CustomModel.from_dict(response["finetune"], self.wait_for_custom_model)

    def wait_for_custom_model(
        self,
        custom_model_id: str,
        timeout: Optional[float] = None,
        interval: float = 60,
    ) -> CustomModel:
        """Wait for custom model training completion.

        Args:
            custom_model_id (str): Custom model id.
            timeout (Optional[float], optional): Wait timeout in seconds, if None - there is no limit to the wait time.
                Defaults to None.
            interval (float, optional): Wait poll interval in seconds. Defaults to 10.

        Raises:
            TimeoutError: wait timed out

        Returns:
            BulkEmbedJob: Custom model.
        """

        return wait_for_job(
            get_job=partial(self.get_custom_model, custom_model_id),
            timeout=timeout,
            interval=interval,
        )

    def _upload_dataset(
        self, content: Iterable[bytes], custom_model_name: str, file_name: str, type: INTERNAL_CUSTOM_MODEL_TYPE
    ) -> str:
        gcs = self._create_signed_url(custom_model_name, file_name, type)
        response = requests.put(gcs["url"], data=content, headers={"content-type": "text/plain"})
        if response.status_code != 200:
            raise CohereError(message=f"Unexpected server error (status {response.status_code}): {response.text}")
        return gcs["gcspath"]

    def _create_signed_url(
        self, custom_model_name: str, file_name: str, type: INTERNAL_CUSTOM_MODEL_TYPE
    ) -> TypedDict("gcsData", {"url": str, "gcspath": str}):
        json = {"finetuneName": custom_model_name, "fileName": file_name, "finetuneType": type}
        return self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneUploadSignedURL", method="POST", json=json)

    def get_custom_model(self, custom_model_id: str) -> CustomModel:
        """Get a custom model by id.

        Args:
            custom_model_id (str): custom model id
        Returns:
            CustomModel: the custom model
        """
        json = {"finetuneID": custom_model_id}
        response = self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetune", method="POST", json=json)
        return CustomModel.from_dict(response["finetune"], self.wait_for_custom_model)

    def get_custom_model_by_name(self, name: str) -> CustomModel:
        """Get a custom model by name.

        Args:
            name (str): custom model name
        Returns:
            CustomModel: the custom model
        """
        json = {"name": name}
        response = self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneByName", method="POST", json=json)
        return CustomModel.from_dict(response["finetune"], self.wait_for_custom_model)

    def get_custom_model_metrics(self, custom_model_id: str) -> List[ModelMetric]:
        """Get a custom model's training metrics by id

        Args:
            custom_model_id (str): custom model id
        Returns:
            List[ModelMetric]: a list of model metrics
        """
        json = {"finetuneID": custom_model_id}
        response = self._request(f"{cohere.CUSTOM_MODEL_URL}/GetFinetuneMetrics", method="POST", json=json)
        return [ModelMetric.from_dict(metric) for metric in response["metrics"]]

    def list_custom_models(
        self,
        statuses: Optional[List[CUSTOM_MODEL_STATUS]] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        order_by: Optional[Literal["asc", "desc"]] = None,
    ) -> List[CustomModel]:
        """List custom models of your organization. Limit is 50.

        Args:
            statuses (CUSTOM_MODEL_STATUS, optional): search for fintunes which are in one of these states
            before (datetime, optional): search for custom models that were created before this timestamp
            after (datetime, optional): search for custom models that were created after this timestamp
            order_by (Literal["asc", "desc"], optional): sort custom models by created at, either asc or desc
        Returns:
            List[CustomModel]: a list of custom models.
        """
        if before:
            before = before.replace(tzinfo=before.tzinfo or timezone.utc)
        if after:
            after = after.replace(tzinfo=after.tzinfo or timezone.utc)
        internal_statuses = []
        if statuses:
            for status in statuses:
                internal_statuses.append(CUSTOM_MODEL_INTERNAL_STATUS_MAPPING[status])
        json = {
            "query": {
                "statuses": internal_statuses,
                "before": before.isoformat(timespec="seconds") if before else None,
                "after": after.isoformat(timespec="seconds") if after else None,
                "orderBy": order_by,
            }
        }

        response = self._request(f"{cohere.CUSTOM_MODEL_URL}/ListFinetunes", method="POST", json=json)
        return [CustomModel.from_dict(r, self.wait_for_custom_model) for r in response["finetunes"]]

    def create_connector(
        self,
        name: str,
        url: str,
        active: bool = True,
        continue_on_failure: bool = False,
        excludes: Optional[List[str]] = None,
        oauth: Optional[dict] = None,
        service_auth: Optional[dict] = None,
    ) -> Connector:
        """Creates a Connector with the provided information

        Args:
            name (str): The name of your connector
            url (str): The URL of the connector that will be used to search for documents
            active (bool): (optional) Whether the connector is active or not
            continue_on_failure (bool): (optional) Whether a chat request should continue or not if the request to this connector fails
            excludes (List[str]): (optional) A list of fields to exclude from the prompt (fields remain in the document)
            oauth (dict): (optional) The OAuth 2.0 configuration for the connector.
            service_auth: (dict): (optional) The service to service authentication configuration for the connector
        Returns:
            Connector: Connector object.
        """
        json = {
            "name": name,
            "url": url,
            "active": active,
            "continue_on_failure": continue_on_failure,
        }
        if oauth is not None:
            json["oauth"] = oauth

        if service_auth is not None:
            json["service_auth"] = service_auth

        if excludes is not None:
            json["excludes"] = excludes

        create_response = self._request(cohere.CONNECTOR_URL, json=json)
        return self.get_connector(id=create_response["connector"]["id"])

    def update_connector(
        self,
        id: str,
        name: Optional[str] = None,
        url: Optional[str] = None,
        active: Optional[bool] = None,
        continue_on_failure: Optional[bool] = None,
        excludes: Optional[List[str]] = None,
        oauth: Optional[dict] = None,
        service_auth: Optional[dict] = None,
    ) -> Connector:
        """Updates a Connector with the provided id

        Args:
            id (str): The ID of the connector you wish to update.
            name (str): (optional) The name of your connector
            url (str): (optional) The URL of the connector that will be used to search for documents
            active (bool): (optional) Whether the connector is active or not
            continue_on_failure (bool): (optional) Whether a chat request should continue or not if the request to this connector fails
            excludes (List[str]): (optional) A list of fields to exclude from the prompt (fields remain in the document)
            oauth (dict): (optional) The OAuth 2.0 configuration for the connector.
            service_auth: (dict): (optional) The service to service authentication configuration for the connector
        Returns:
            Connector: Connector object.
        """
        if not id:
            raise CohereError(message="id must not be empty")
        json = {}
        if url is not None:
            json["url"] = url

        if active is not None:
            json["active"] = active

        if continue_on_failure is not None:
            json["continue_on_failure"] = continue_on_failure

        if name is not None:
            json["name"] = name

        if oauth is not None:
            json["oauth"] = oauth

        if service_auth is not None:
            json["service_auth"] = service_auth

        if excludes is not None:
            json["excludes"] = excludes

        update_response = self._request(f"{cohere.CONNECTOR_URL}/{id}", method="PATCH", json=json)
        return self.get_connector(id=update_response["connector"]["id"])

    def get_connector(self, id: str) -> Connector:
        """Returns a Connector given an id

        Args:
            id (str): The id of your connector

        Returns:
            Connector: Connector object.
        """
        if not id:
            raise CohereError(message="id must not be empty")
        response = self._request(f"{cohere.CONNECTOR_URL}/{id}", method="GET")
        return Connector.from_dict(response["connector"])

    def list_connectors(self, limit: int = None, offset: int = None) -> List[Connector]:
        """Returns a list of your Connectors

        Args:
            limit (int): (optional) The max number of connectors to return
            offset (int): (optional) The number of connectors to offset by

        Returns:
            List[Connector]: List of Connector objects.
        """
        param_dict = {}

        if limit is not None:
            param_dict["limit"] = limit

        if offset is not None:
            param_dict["offset"] = offset
        response = self._request(f"{cohere.CONNECTOR_URL}", method="GET", params=param_dict)
        return [Connector.from_dict(r) for r in (response.get("connectors") or [])]

    def delete_connector(self, id: str) -> None:
        """Deletes a Connector given an id

        Args:
            id (str): The id of your connector
        """
        if not id:
            raise CohereError(message="id must not be empty")
        self._request(f"{cohere.CONNECTOR_URL}/{id}", method="DELETE")

    def oauth_authorize_connector(self, id: str, after_token_redirect: str = None) -> str:
        """Returns a URL which when navigated to will start the OAuth 2.0 flow.

        Args:
            id (str): The id of your connector

        Returns:
            str: A URL that starts the OAuth 2.0 flow.
        """
        if not id:
            raise CohereError(message="id must not be empty")

        param_dict = {}

        if after_token_redirect is not None:
            param_dict["after_token_redirect"] = after_token_redirect

        response = self._request(f"{cohere.CONNECTOR_URL}/{id}/oauth/authorize", method="GET", params=param_dict)
        return response["redirect_url"]
