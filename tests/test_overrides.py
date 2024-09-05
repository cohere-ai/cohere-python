import unittest
from contextlib import redirect_stderr
import logging


from cohere import EmbedByTypeResponseEmbeddings

LOGGER = logging.getLogger(__name__)

class TestClient(unittest.TestCase):

    def test_float_alias(self) -> None:
        embeds = EmbedByTypeResponseEmbeddings(float_=[[1.0]])
        self.assertEqual(embeds.float_, [[1.0]])
        self.assertEqual(embeds.float, [[1.0]])  # type: ignore
