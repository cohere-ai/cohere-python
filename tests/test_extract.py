import unittest
import cohere
from cohere.extract import Example, Entity, Extractions
from utils import get_api_key

co = cohere.Client(get_api_key())


class TestExtract(unittest.TestCase):
    def test_success(self):
        examples = [Example(
            text="hello my name is John, and I like to play ping pong",
            entities=[Entity(type="Name", value="John")])]
        texts = ["hello Roberta, how are you doing today?"]

        extractions = co.extract('small', examples, texts)

        self.assertIsInstance(extractions, Extractions)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertEqual(extractions[0].entities[0].type, "Name")
        self.assertEqual(extractions[0].entities[0].value, "Roberta")

    def test_empty_text(self):
        with self.assertRaises(cohere.CohereError):
            co.extract(
                'small', examples=[Example(
                    text="hello my name is John, and I like to play ping pong",
                    entities=[Entity(type="Name", value="John")])],
                texts=[""])

    def test_empty_entities(self):
        with self.assertRaises(cohere.CohereError):
            co.extract(
                'large', examples=[Example(
                    text="hello my name is John, and I like to play ping pong",
                    entities=[])],
                texts=["hello Roberta, how are you doing today?"])

    def test_varying_amount_of_entities(self):
        examples = [
            Example(
                text="the bananas are red",
                entities=[Entity(type="fruit", value="bananas"), Entity(type="color", value="red")]),
            Example(
                text="i love the color blue",
                entities=[Entity(type="color", value="blue")]),
            Example(
                text="i love apples",
                entities=[Entity(type="fruit", value="apple")]),
            Example(
                text="purple is my favorite color",
                entities=[Entity(type="color", value="purple")]),
            Example(
                text="wow, that apple is green?",
                entities=[Entity(type="fruit", value="apple"), Entity(type="color", value="green")])]
        texts = ["i love bananas", "my favorite color is yellow", "i love green apples"]

        extractions = co.extract('medium', examples, texts)

        self.assertIsInstance(extractions, Extractions)
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
            Example(
                text="hello my name is John, and I like to play ping pong",
                entities=[Entity(type="Name", value="John"), Entity(type="Game", value="ping pong")]),
            Example(
                text="greetings, I'm Roberta and I like to play golf",
                entities=[Entity(type="Name", value="Roberta"), Entity(type="Game", value="golf")]),
            Example(
                text="let me introduce myself, my name is Tina and I like to play baseball",
                entities=[Entity(type="Name", value="Tina"), Entity(type="Game", value="baseball")])]
        texts = ["hi, my name is Charlie and I like to play basketball", "hello, I'm Olivia and I like to play soccer"]

        extractions = co.extract('medium', examples, texts)

        self.assertEqual(len(extractions), 2)
        self.assertIsInstance(extractions, Extractions)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[1].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertIsInstance(extractions[1].entities, list)
        self.assertEqual(len(extractions[0].entities), 2)
        self.assertEqual(len(extractions[1].entities), 2)

    def test_no_entities(self):
        examples = [
            Example(
                text="hello my name is John, and I like to play ping pong",
                entities=[Entity(type="Name", value="John"), Entity(type="Game", value="ping pong")]),
            Example(
                text="greetings, I'm Roberta and I like to play golf",
                entities=[Entity(type="Name", value="Roberta"), Entity(type="Game", value="golf")]),
            Example(
                text="let me introduce myself, my name is Tina and I like to play baseball",
                entities=[Entity(type="Name", value="Tina"), Entity(type="Game", value="baseball")])]
        texts = ["hi, my name is Charlie and I like to play basketball", "hello!"]

        extractions = co.extract('medium', examples, texts)

        self.assertEqual(len(extractions), 2)
        self.assertIsInstance(extractions, Extractions)
        self.assertIsInstance(extractions[0].text, str)
        self.assertIsInstance(extractions[1].text, str)
        self.assertIsInstance(extractions[0].entities, list)
        self.assertIsInstance(extractions[1].entities, list)
        self.assertEqual(len(extractions[0].entities), 2)
        self.assertEqual(len(extractions[1].entities), 0)
