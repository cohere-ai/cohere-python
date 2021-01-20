import json
from urllib.parse import urljoin
import requests

COHERE_API_URL = "https://api.cohere.ai"
SAMPLE_URL = "sample"
SIMILARITY_URL = "similarity"
EMBED_URL = "embed"
CHOOSE_BEST_URL = "choose-best"

class CohereClient:
  def __init__(self, api_key):
    self.api_key = api_key
    self.model = None

  def sample(self, model, prompt, num_tokens=20, num_samples=1, temperature=1):
    json_body = json.dumps({
        "prompt": prompt,
        "num_samples": num_samples,
        "max_tokens": num_tokens,
        "temperature": temperature,
      })
    response = self.__request(json_body, SAMPLE_URL, model)
    return response["text"]

  def similarity(self, model, anchor, targets):
    json_body = json.dumps({
        "anchor": anchor,
        "targets": targets,
      })
    response = self.__request(json_body, SIMILARITY_URL, model)
    return response["similarities"]

  def embed(self, model, texts):
    json_body = json.dumps({
        "texts": texts,
      })
    response = self.__request(json_body, EMBED_URL, model)
    return response["embeddings"]

  def choose_best(self, model, query, options):
    json_body = json.dumps({
        "query": query,
        "options": options,
      })
    response = self.__request(json_body, CHOOSE_BEST_URL, model)
    return response["rankedOptions"]

  def __request(self, json_body, endpoint, model):
    headers = {
      'Authorization': 'BEARER {}'.format(self.api_key),
      'Content-Type': 'application/json'
    }
    url = urljoin(COHERE_API_URL, model + "/" + endpoint)
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
