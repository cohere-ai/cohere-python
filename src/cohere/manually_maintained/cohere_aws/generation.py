from .response import CohereObject
from .mode import Mode
from typing import List, Optional, NamedTuple, Generator, Dict, Any
import json


class TokenLikelihood(CohereObject):
    def __init__(self, token: str, likelihood: float) -> None:
        self.token = token
        self.likelihood = likelihood


class Generation(CohereObject):
    def __init__(self,
                 text: str,
                 token_likelihoods: List[TokenLikelihood]) -> None:
        self.text = text
        self.token_likelihoods = token_likelihoods


class Generations(CohereObject):
    def __init__(self,
                 generations: List[Generation]) -> None:
        self.generations = generations
        self.iterator = iter(generations)

    @classmethod
    def from_dict(cls, response: Dict[str, Any]) -> List[Generation]:
        generations: List[Generation] = []
        for gen in response['generations']:
            token_likelihoods = None

            if 'token_likelihoods' in gen:
                token_likelihoods = []
                for likelihoods in gen['token_likelihoods']:
                    if 'likelihood' in likelihoods:
                        token_likelihood = likelihoods['likelihood']
                    else:
                        token_likelihood = None
                    token_likelihoods.append(TokenLikelihood(
                        likelihoods['token'], token_likelihood))
            generations.append(Generation(gen['text'], token_likelihoods))
        return cls(generations)

    def __iter__(self) -> iter:
        return self.iterator

    def __next__(self) -> next:
        return next(self.iterator)


StreamingText = NamedTuple("StreamingText",
                           [("index", Optional[int]),
                            ("text", str),
                            ("is_finished", bool)])


class StreamingGenerations(CohereObject):
    def __init__(self, stream, mode):
        self.stream = stream
        self.id = None
        self.generations = None
        self.finish_reason = None
        self.bytes = bytearray()

        if mode == Mode.SAGEMAKER:
            self.payload_key = "PayloadPart"
            self.bytes_key = "Bytes"
        elif mode == Mode.BEDROCK:
            self.payload_key = "chunk"
            self.bytes_key = "bytes"
        else:
            raise CohereError("Unsupported mode")

    def _make_response_item(self, streaming_item) -> Optional[StreamingText]:
        is_finished = streaming_item.get("is_finished")

        if not is_finished:
            index = streaming_item.get("index", 0)
            text = streaming_item.get("text")
            if text is None:
                return None
            return StreamingText(
                text=text, is_finished=is_finished, index=index)

        self.finish_reason = streaming_item.get("finish_reason")
        generation_response = streaming_item.get("response")

        if generation_response is None:
            return None

        self.id = generation_response.get("id")
        self.generations = Generations.from_dict(generation_response)
        return None

    def __iter__(self) -> Generator[StreamingText, None, None]:
        for payload in self.stream:
            self.bytes.extend(payload[self.payload_key][self.bytes_key])
            try:
                item = self._make_response_item(json.loads(self.bytes))
            except json.decoder.JSONDecodeError:
                # payload contained only a partion JSON object
                continue

            self.bytes = bytearray()
            if item is not None:
                yield item
