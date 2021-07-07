import os

API_KEY = os.getenv("CO_API_KEY")

import cohere

co = cohere.CohereClient(API_KEY)


### GENERATE
prediction = co.generate(
            model="baseline-shrimp",
            prompt="co:here",
            max_tokens=10)
print('prediction: {}'.format(prediction.text))


### EMBED
embeddings = co.embed(
            model="baseline-shrimp",
            texts=["co:here", "cohere"])
print('Embedding of `co:here`: {}'.format(embeddings.embeddings[0]))

for em in embeddings:
      print(em)


### SIMILARITY
similarities = co.similarity(
            model="baseline-shrimp",
           	anchor="cohere ai",
            targets=["co:here", "cohere"])
print('Similarity value of `co:here`: {}'.format(similarities.similarities[0]))

for sim in similarities:
      print(sim)


### CHOOSE BEST
options = co.choose_best(
            model="baseline-shrimp",
            query="hello {}",
            options=["world", "cohere"],
            mode="APPEND_OPTION")
print('first option is `world`, with likelihood value of {}'.format(options.likelihoods[0]))
print('Selected mode was {}'.format(options.mode))

for op in options:
      print(op)


### LIKELIHOOD
likelihood = co.likelihood(
            model="baseline-shrimp",
            text="hello, my name is johnny SURPRISE")
print('likelihood of text is {}'.format(likelihood.likelihood))
print('token likelihoods are: (first token has no likelihood)')
for token in likelihood.token_likelihoods:
      print(token['token'], token.get('likelihood', ''))


### ERRORS
try:
	predictions = co.generate(
            model="fake-model",
            prompt="co:here",
            max_tokens=10)
except cohere.CohereError as e:
	print(e) # could not find model with name fake-model

try:
	predictions = co.generate(
            model="baseline-shrimp",
            prompt="",
            max_tokens=10)
except cohere.CohereError as e:
	print(e) # prompt length must be greater than 0
