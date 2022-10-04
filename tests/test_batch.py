import unittest

import test_generate, test_embed

def double_pormpt(kwargs):
    kwargs['prompts'] = [kwargs['prompt'], kwargs['prompt']]
    del kwargs['prompt']
    return kwargs

test_generate.co.generate = lambda **kwargs: test_generate.co.generate_batch(**double_pormpt(kwargs))[0]


class TestBatch(unittest.TestCase):

    def test_generate_batch(self):
        cls = test_generate.TestGenerate()
        for method in dir(cls):
            if method.startswith('test_') and method != 'test_preset_success':
                getattr(cls, method)()
