import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("CO_API_KEY")

import cohere

co = cohere.CohereClient(API_KEY)

predictions = co.generate(
            model="baseline-124m",
            prompt="co:here",
            num_tokens=10,
            num_generations=3)
print('First prediction: {}'.format(predictions[0]))

embeddings = co.embed(
            model="baseline-embed",
            texts=["co:here", "cohere"])

similarities = co.similarity(
            model="baseline-similarity",
           	anchor="cohere ai",
            targets=["co:here", "cohere"])
print('Similarity value of `co:here`: {}'.format(similarities[0]))

best_options = co.choose_best(
            model="baseline-likelihood",
            query="hello {}",
            options=["world", "cohere"])
print('Best option is `{}`, with likelihood value of {}'.format(best_options['rankedOptions'][0]['option'], best_options['rankedOptions'][0]['likelihood']))
print('Selected mode was {}'.format(best_options['mode']))

try:
	predictions = co.generate(
            model="fake-model",
            prompt="co:here",
            num_tokens=10,
            num_generations=3)
except cohere.CohereError as e:
	print(e) # could not find model with name fake-model

try:
	predictions = co.generate(
            model="baseline-124m",
            prompt="",
            num_tokens=10,
            num_generations=3)
except cohere.CohereError as e:
	print(e) # prompt length must be greater than 0
