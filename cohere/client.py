
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
    return self.__request(json_body, SAMPLE_URL, model)

  def similarity(self, model, anchor, targets):
    json_body = json.dumps({
        "anchor": anchor,
        "targets": targets,
      })
    return self.__request(json_body, SIMILARITY_URL, model)

  def embed(self, model, texts):
    json_body = json.dumps({
        "texts": texts,
      })
    return self.__request(json_body, EMBED_URL, model)

  def choose_best(self, model, query, options):
    json_body = json.dumps({
        "query": query,
        "options": options,
      })
    return self.__request(json_body, CHOOSE_BEST_URL, model)

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
