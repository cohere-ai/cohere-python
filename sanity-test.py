import os

API_KEY = os.getenv("CO_API_KEY")

import cohere

co = cohere.CohereClient(API_KEY)

predictions = co.generate(
            model="baseline-1b",
            prompt="co:here",
            max_tokens=10)
print('First prediction: {}'.format(predictions[0]))

embeddings = co.embed(
            model="baseline-124m",
            texts=["co:here", "cohere"])

similarities = co.similarity(
            model="baseline-124m",
           	anchor="cohere ai",
            targets=["co:here", "cohere"])
print('Similarity value of `co:here`: {}'.format(similarities[0]))

options = co.choose_best(
            model="baseline-355m",
            query="hello {}",
            options=["world", "cohere"])
print('first option is `world`, with likelihood value of {}'.format(options['likelihoods'][0]))
print('Selected mode was {}'.format(options['mode']))

try:
	predictions = co.generate(
            model="fake-model",
            prompt="co:here",
            max_tokens=10)
except cohere.CohereError as e:
	print(e) # could not find model with name fake-model

try:
	predictions = co.generate(
            model="baseline-124m",
            prompt="",
            max_tokens=10)
except cohere.CohereError as e:
	print(e) # prompt length must be greater than 0
