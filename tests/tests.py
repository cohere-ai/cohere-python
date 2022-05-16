import os
import unittest
import cohere
import string
import random
from cohere.classify import Example
from cohere.extract import Entity, Example as ExtractExample

API_KEY = os.getenv('CO_API_KEY')
assert type(API_KEY)
co = cohere.Client(str(API_KEY))

letters = string.ascii_lowercase


def random_word():
    return ''.join(random.choice(letters) for _ in range(10))


def random_sentence(num_words):
    sentence = ''

    for _ in range(num_words):
        sentence += random_word() + ' '

    return sentence


def random_texts(num_texts, num_words_per_sentence=50):
    arr = []

    for _ in range(num_texts):
        arr.append(random_sentence(num_words_per_sentence))

    return arr


class TestModel(unittest.TestCase):
    def test_invalid_model(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(
                model='this-better-not-exist',
                prompt='co:here',
                max_tokens=1)

    def test_no_version_works(self):
        cohere.Client(API_KEY).generate(
            model='small',
            prompt='co:here',
            max_tokens=1)

    def test_invalid_version_fails(self):
        with self.assertRaises(cohere.CohereError):
            cohere.Client(API_KEY, 'fake').generate(
                model='small',
                prompt='co:here',
                max_tokens=1)

    def test_invalid_key(self):
        with self.assertRaises(cohere.CohereError):
            _ = cohere.Client('invalid')


class TestGenerate(unittest.TestCase):
    def test_success(self):
        prediction = co.generate(
            model='small',
            prompt='co:here',
            max_tokens=1)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsNone(prediction.generations[0].token_likelihoods)
        self.assertEqual(prediction.return_likelihoods, 'NONE')

    def test_return_likelihoods_generation(self):
        prediction = co.generate(
            model='small',
            prompt='co:here',
            max_tokens=1,
            return_likelihoods='GENERATION')
        self.assertTrue(prediction.generations[0].token_likelihoods)
        self.assertTrue(prediction.generations[0].token_likelihoods[0].token)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'GENERATION')

    def test_return_likelihoods_all(self):
        prediction = co.generate(
            model='small',
            prompt='hi',
            max_tokens=1,
            return_likelihoods='ALL')
        self.assertEqual(len(prediction.generations[0].token_likelihoods), 2)
        self.assertIsNotNone(prediction.generations[0].likelihood)
        self.assertEqual(prediction.return_likelihoods, 'ALL')

    def test_invalid_temp(self):
        with self.assertRaises(cohere.CohereError):
            co.generate(
                model='large',
                prompt='hi',
                max_tokens=1,
                temperature=-1)


class TestEmbed(unittest.TestCase):
    def test_success(self):
        prediction = co.embed(
            model='small',
            texts=['co:here', 'cohere'])
        self.assertEqual(len(prediction.embeddings), 2)
        self.assertIsInstance(prediction.embeddings[0], list)
        self.assertIsInstance(prediction.embeddings[1], list)
        self.assertEqual(len(prediction.embeddings[0]), 1024)
        self.assertEqual(len(prediction.embeddings[1]), 1024)

    def test_success_multiple_batches(self):
        prediction = co.embed(
            model='small',
            texts=['co:here', 'cohere', 'embed', 'python', 'golang', 'typescript', 'rust?', 'ai', 'nlp', 'neural'])
        self.assertEqual(len(prediction.embeddings), 10)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_longer_multiple_batches_unaligned_batch(self):
        prediction = co.embed(
            model='small',
            texts=[
                'co:here', 'cohere', 'embed', 'python', 'golang',
                'typescript', 'rust?',  'ai',  'nlp', 'neural', 'nets'
            ])
        self.assertEqual(len(prediction.embeddings), 11)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_longer_multiple_batches(self):
        prediction = co.embed(
            model='small',
            texts=['co:here', 'cohere', 'embed', 'python', 'golang'] * 200)
        self.assertEqual(len(prediction.embeddings), 200*5)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_multiple_batches_in_order(self):
        textAll = []
        predictionsExpected = []

        for _ in range(3):
            text_batch = random_texts(cohere.COHERE_EMBED_BATCH_SIZE)
            prediction = co.embed(
                model='small',
                texts=text_batch)
            textAll.extend(text_batch)
            predictionsExpected.extend(prediction)
        predictionsActual = co.embed(model='small', texts=textAll)
        for predictionExpected, predictionActual in zip(predictionsExpected, list(predictionsActual)):
            for elementExpected, elementAcutal in zip(predictionExpected, predictionActual):
                self.assertAlmostEqual(elementExpected, elementAcutal, places=1)

    def test_invalid_texts(self):
        with self.assertRaises(cohere.CohereError):
            co.embed(
                model='small',
                texts=[''])


class TestClassify(unittest.TestCase):
    def test_success(self):
        prediction = co.classify('medium', ['purple'], [
            Example('apple', 'fruit'),
            Example('banana', 'fruit'),
            Example('cherry', 'fruit'),
            Example('watermelon', 'fruit'),
            Example('kiwi', 'fruit'),
            Example('red', 'color'),
            Example('blue', 'color'),
            Example('green', 'color'),
            Example('yellow', 'color'),
            Example('magenta', 'color')
        ])
        self.assertIsInstance(prediction.classifications, list)
        self.assertIsInstance(prediction.classifications[0].input, str)
        self.assertIsInstance(prediction.classifications[0].prediction, str)
        self.assertIsInstance(prediction.classifications[0].confidence[0].confidence, (int, float))
        self.assertIsInstance(prediction.classifications[0].confidence[0].label, str)
        self.assertIsInstance(prediction.classifications[0].confidence[1].confidence, (int, float))
        self.assertIsInstance(prediction.classifications[0].confidence[1].label, str)
        self.assertEqual(len(prediction.classifications), 1)
        self.assertEqual(prediction.classifications[0].prediction, 'color')

    def test_empty_inputs(self):
        with self.assertRaises(cohere.CohereError):
            co.classify(
                'medium', [], [
                    Example('apple', 'fruit'),
                    Example('banana', 'fruit'),
                    Example('cherry', 'fruit'),
                    Example('watermelon', 'fruit'),
                    Example('kiwi', 'fruit'),

                    Example('red', 'color'),
                    Example('blue', 'color'),
                    Example('green', 'color'),
                    Example('yellow', 'color'),
                    Example('magenta', 'color')])

    def test_success_multi_input(self):
        prediction = co.classify('medium', ['purple', 'mango'], [
            Example('apple', 'fruit'),
            Example('banana', 'fruit'),
            Example('cherry', 'fruit'),
            Example('watermelon', 'fruit'),
            Example('kiwi', 'fruit'),

            Example('red', 'color'),
            Example('blue', 'color'),
            Example('green', 'color'),
            Example('yellow', 'color'),
            Example('magenta', 'color')])
        self.assertEqual(prediction.classifications[0].prediction, 'color')
        self.assertEqual(prediction.classifications[1].prediction, 'fruit')
        self.assertEqual(len(prediction.classifications), 2)

    def test_success_all_fields(self):
        prediction = co.classify('medium', ['mango', 'purple'], [
            Example('apple', 'fruit'),
            Example('banana', 'fruit'),
            Example('cherry', 'fruit'),
            Example('watermelon', 'fruit'),
            Example('kiwi', 'fruit'),

            Example('red', 'color'),
            Example('blue', 'color'),
            Example('green', 'color'),
            Example('yellow', 'color'),
            Example('magenta', 'color')
        ], 'this is a classifier to determine if a word is a fruit of a color', 'This is a')
        self.assertEqual(prediction.classifications[0].prediction, 'fruit')
        self.assertEqual(prediction.classifications[1].prediction, 'color')


class TestExtract(unittest.TestCase):
    def test_success(self):
        examples = [ExtractExample(
            text="hello my name is John, and I like to play ping pong",
            entities=[Entity(type="Name", value="John")])]
        texts = ["hello Roberta, how are you doing today?"]

        extractions = co.extract('small', examples, texts)

        self.assertIsInstance(extractions, list)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertEqual(extractions[0].entities[0].type, "Name")
        self.assertEqual(extractions[0].entities[0].value, "Roberta")

    def test_empty_text(self):
        with self.assertRaises(cohere.CohereError):
            co.extract(
                'small', examples=[ExtractExample(
                    text="hello my name is John, and I like to play ping pong",
                    entities=[Entity(type="Name", value="John")])],
                texts=[""])

    def test_empty_entities(self):
        with self.assertRaises(cohere.CohereError):
            co.extract(
                'large', examples=[ExtractExample(
                    text="hello my name is John, and I like to play ping pong",
                    entities=[])],
                texts=["hello Roberta, how are you doing today?"])

    def test_varying_amount_of_entities(self):
        examples = [
            ExtractExample(
                text="the bananas are red",
                entities=[Entity(type="fruit", value="bananas"), Entity(type="color", value="red")]),
            ExtractExample(
                text="i love the color blue",
                entities=[Entity(type="color", value="blue")]),
            ExtractExample(
                text="i love apples",
                entities=[Entity(type="fruit", value="apple")]),
            ExtractExample(
                text="purple is my favorite color",
                entities=[Entity(type="color", value="purple")]),
            ExtractExample(
                text="wow, that apple is green?",
                entities=[Entity(type="fruit", value="apple"), Entity(type="color", value="green")])]
        texts = ["i love bananas", "my favorite color is yellow", "i love green apples"]

        extractions = co.extract('medium', examples, texts)

        self.assertIsInstance(extractions, list)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[1].text, str)
        self.assertIsInstance(extractions[2].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertIsInstance(extractions[1].entities, list)
        self.assertIsInstance(extractions[2].entities, list)
        self.assertEqual(len(extractions[0].entities), 1)
        self.assertEqual(len(extractions[1].entities), 1)
        self.assertEqual(len(extractions[2].entities), 2)

    def test_many_examples_and_multiple_texts(self):
        examples = [
            ExtractExample(
                text="hello my name is John, and I like to play ping pong",
                entities=[Entity(type="Name", value="John"), Entity(type="Game", value="ping pong")]),
            ExtractExample(
                text="greetings, I'm Roberta and I like to play golf",
                entities=[Entity(type="Name", value="Roberta"), Entity(type="Game", value="golf")]),
            ExtractExample(
                text="let me introduce myself, my name is Tina and I like to play baseball",
                entities=[Entity(type="Name", value="Tina"), Entity(type="Game", value="baseball")])]
        texts = ["hi, my name is Charlie and I like to play basketball", "hello, I'm Olivia and I like to play soccer"]

        extractions = co.extract('medium', examples, texts)

        self.assertEqual(len(extractions), 2)
        self.assertIsInstance(extractions, list)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[1].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertIsInstance(extractions[1].entities, list)
        self.assertEqual(len(extractions[0].entities), 2)
        self.assertEqual(len(extractions[1].entities), 2)

    def test_no_entities(self):
        examples = [
            ExtractExample(
                text="hello my name is John, and I like to play ping pong",
                entities=[Entity(type="Name", value="John"), Entity(type="Game", value="ping pong")]),
            ExtractExample(
                text="greetings, I'm Roberta and I like to play golf",
                entities=[Entity(type="Name", value="Roberta"), Entity(type="Game", value="golf")]),
            ExtractExample(
                text="let me introduce myself, my name is Tina and I like to play baseball",
                entities=[Entity(type="Name", value="Tina"), Entity(type="Game", value="baseball")])]
        texts = ["hi, my name is Charlie and I like to play basketball", "hello!"]

        extractions = co.extract('medium', examples, texts)

        self.assertEqual(len(extractions), 2)
        self.assertIsInstance(extractions, list)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[1].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertIsInstance(extractions[1].entities, list)
        self.assertEqual(len(extractions[0].entities), 2)
        self.assertEqual(len(extractions[1].entities), 0)


class TestTokenize(unittest.TestCase):
    def test_success(self):
        tokens = co.tokenize('medium', 'tokenize me!')
        self.assertIsInstance(tokens.tokens, list)
        self.assertIsInstance(tokens.length, int)
        self.assertEqual(tokens.length, len(tokens))

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.tokenize(model='medium', text='')


if __name__ == '__main__':
    unittest.main()
