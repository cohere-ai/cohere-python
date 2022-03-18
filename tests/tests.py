import os
import unittest
import cohere
import string
import random
from typing import List
from cohere.classify import Example
from cohere.assert_parameters import assert_parameter, assert_list_parameter

API_KEY = os.getenv('CO_API_KEY')
assert type(API_KEY) != None
co = cohere.Client(str(API_KEY))

letters = string.ascii_lowercase

def random_word():
    return ''.join(random.choice(letters) for _ in range(10))

def random_sentence(num_words):
    sentence = ""

    for _ in range(num_words):
        sentence += random_word() + " "

    return sentence

def random_texts(num_texts, num_words_per_sentence = 50):
    arr = []

    for _ in range(num_texts):
        arr.append(random_sentence(num_words_per_sentence))
    
    return arr

validStr = "test"
validStrList = ["test", "hello", "cohere"]
validEx = Example("a", "b")
validExList = [Example("a", "b"), Example("apple", "fruit"), Example("this movie is great", "positive")]

def exampleFunction(stringParam: str, listParam: List[str], exampleParam: Example, exampleList: List[Example]):
    exampleList = assert_list_parameter(Example, "exampleList", exampleList, "endpoint")
    listParam = assert_list_parameter(str, "listParam", listParam, "endpoint")
    assert_parameter(str, "stringParam", stringParam, "endpoint")
    assert_parameter(Example, "exampleParam", exampleParam, "endpoint")
    return [stringParam, listParam, exampleParam, exampleList]

class TestAssertParameters(unittest.TestCase):
    def test_valid_basic(self):
        returningParameters = exampleFunction(validStr, validStrList, validEx, validExList)
        self.assertEqual(returningParameters, [validStr, validStrList, validEx, validExList])
    def test_invalid_basic_type(self):
        with self.assertRaises(cohere.CohereError):
            exampleFunction(3, validEx, validEx, validExList)

    def test_invalid_basic_list_1(self):
        with self.assertRaises(cohere.CohereError):
            exampleFunction(validStr, ["hello", 3, Example("a", "b")], validEx, validExList)

    def test_invalid_basic_list_2(self):
        with self.assertRaises(cohere.CohereError):
            exampleFunction(validStr, 3, validEx, validExList)
    
    def test_valid_basic_list(self):
        returningParameters = exampleFunction(validStr, "test", validEx, validExList)
        self.assertEqual(returningParameters, [validStr, ["test"], validEx, validExList])

    def test_invalid_object_list(self):
        with self.assertRaises(cohere.CohereError):
            exampleFunction(validStr, validStrList, validEx, [Example("hi", "hello"), "yes", "no"])

    def test_invalid_object(self):
        with self.assertRaises(cohere.CohereError):
            exampleFunction(validStr, validStrList, 0, validExList)
    
    def test_valid_object_list(self):
        returningParameters = exampleFunction(validStr, validStrList, validEx, Example("test", "should be valid"))
        self.assertEqual(returningParameters[3][0].text, "test")
        self.assertEqual(returningParameters[3][0].label, "should be valid")
        
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
            texts=['co:here', 'cohere', "embed", "python", "golang", "typescript", "rust?", "ai", "nlp","neural"])
        self.assertEqual(len(prediction.embeddings), 10)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_longer_multiple_batches_unaligned_batch(self):
        prediction = co.embed(
            model='small',
            texts=['co:here', 'cohere', "embed", "python", "golang", "typescript", "rust?", "ai", "nlp", "neural", "nets"])
        self.assertEqual(len(prediction.embeddings), 11)
        for embed in prediction.embeddings:
            self.assertIsInstance(embed, list)
            self.assertEqual(len(embed), 1024)

    def test_success_longer_multiple_batches(self):
        prediction = co.embed(
            model='small',
            texts=['co:here', 'cohere', "embed", "python", "golang"] * 200)
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
        predictionsActual = co.embed(model='small',texts=textAll)
        for predictionExpected, predictionActual in zip(predictionsExpected, list(predictionsActual)):
            for elementExpected, elementAcutal in zip (predictionExpected, predictionActual):
                self.assertAlmostEqual(elementExpected, elementAcutal, places=1)

    def test_invalid_texts(self):
        with self.assertRaises(cohere.CohereError):
            co.embed(
                model='small',
                texts=[''])

class TestChooseBest(unittest.TestCase):
    def test_success(self):
        prediction = co.choose_best(
            model='large',
            query='Carol picked up a book and walked to the kitchen. She set it down, picked up her glasses and left. This is in the kitchen now: ',
            options=['book', 'glasses', 'dog'],
            mode='APPEND_OPTION')
        self.assertEqual(len(prediction.scores), 3)
        self.assertIsInstance(prediction.scores[0], float)
        self.assertEqual(len(prediction.tokens), 3)
        self.assertIsInstance(prediction.tokens[0], list)
        self.assertIsInstance(prediction.tokens[0][0], str)
        self.assertEqual(len(prediction.token_log_likelihoods), 3)
        self.assertIsInstance(prediction.token_log_likelihoods[0][0], float)

    def test_empty_options(self):
        with self.assertRaises(cohere.CohereError):
            co.choose_best(
                model='large',
                query='the best book in the world is',
                options=[],
                mode='APPEND_OPTION')

class TestClassify(unittest.TestCase):
    def test_success(self):
        prediction = co.classify('medium', ["purple"], 
        [Example("apple", "fruit"), Example("banana", "fruit"), Example("cherry", "fruit"), Example("watermelon", "fruit"), Example("kiwi", "fruit"), 
        Example("red", "color"), Example("blue", "color"), Example("green", "color"), Example("yellow", "color"), Example("magenta", "color")])
        self.assertIsInstance(prediction.classifications, list)
        self.assertIsInstance(prediction.classifications[0].input, str)
        self.assertIsInstance(prediction.classifications[0].prediction, str)
        self.assertIsInstance(prediction.classifications[0].confidence[0].confidence, float)
        self.assertIsInstance(prediction.classifications[0].confidence[0].label, str)
        self.assertIsInstance(prediction.classifications[0].confidence[1].confidence, float)
        self.assertIsInstance(prediction.classifications[0].confidence[1].label, str)
        self.assertEqual(len(prediction.classifications), 1)
        self.assertEqual(prediction.classifications[0].prediction, "color")

    def test_empty_inputs(self):
        with self.assertRaises(cohere.CohereError):
            classifications = co.classify(
                'medium', [], 
                [Example("apple", "fruit"), Example("banana", "fruit"), Example("cherry", "fruit"), Example("watermelon", "fruit"), Example("kiwi", "fruit"), 
                Example("red", "color"), Example("blue", "color"), Example("green", "color"), Example("yellow", "color"), Example("magenta", "color")])

    def test_success_multi_input(self):
        prediction = co.classify('medium', ["purple", "mango"],
        [Example("apple", "fruit"), Example("banana", "fruit"), Example("cherry", "fruit"), Example("watermelon", "fruit"), Example("kiwi", "fruit"), 
        Example("red", "color"), Example("blue", "color"), Example("green", "color"), Example("yellow", "color"), Example("magenta", "color")])
        self.assertEqual(prediction.classifications[0].prediction, "color")
        self.assertEqual(prediction.classifications[1].prediction, "fruit")
        self.assertEqual(len(prediction.classifications), 2)

    def test_success_all_fields(self):
        prediction = co.classify('medium', ["mango", "purple"],
        [Example("apple", "fruit"), Example("banana", "fruit"), Example("cherry", "fruit"), Example("watermelon", "fruit"), Example("kiwi", "fruit"), 
        Example("red", "color"), Example("blue", "color"), Example("green", "color"), Example("yellow", "color"), Example("magenta", "color")], 
        "this is a classifier to determine if a word is a fruit of a color", "This is a")
        self.assertEqual(prediction.classifications[0].prediction, "fruit")
        self.assertEqual(prediction.classifications[1].prediction, "color")


class TestTokenize(unittest.TestCase):
    def test_success(self):
        tokens = co.tokenize('medium', 'tokenize me!')
        self.assertIsInstance(tokens.tokens, list)
        self.assertIsInstance(tokens.length, int)
        self.assertEqual(tokens.length, len(tokens))

    def test_invalid_text(self):
        with self.assertRaises(cohere.CohereError):
            co.tokenize(
                model='medium',
                text='')

if __name__ == '__main__':
    unittest.main()
