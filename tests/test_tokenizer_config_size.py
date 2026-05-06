import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Stub out the `tokenizers` C-extension so the module can be imported in CI
# without the native library present.
if "tokenizers" not in sys.modules:
    tokenizers_stub = types.ModuleType("tokenizers")
    tokenizers_stub.Tokenizer = object  # type: ignore[attr-defined]
    sys.modules["tokenizers"] = tokenizers_stub

from cohere.manually_maintained.tokenizers import _get_tokenizer_config_size


class TestGetTokenizerConfigSize(unittest.TestCase):
    def _make_head_response(self, headers: dict) -> MagicMock:
        resp = MagicMock()
        resp.headers = headers
        return resp

    def test_content_length_header(self) -> None:
        with patch("requests.head", return_value=self._make_head_response({"Content-Length": "2097152"})):
            size = _get_tokenizer_config_size("https://example.com/tokenizer.json")
        self.assertAlmostEqual(size, 2.0)

    def test_goog_stored_content_length_header(self) -> None:
        with patch("requests.head", return_value=self._make_head_response({"x-goog-stored-content-length": "1048576"})):
            size = _get_tokenizer_config_size("https://example.com/tokenizer.json")
        self.assertAlmostEqual(size, 1.0)

    def test_goog_header_takes_priority_over_content_length(self) -> None:
        with patch(
            "requests.head",
            return_value=self._make_head_response(
                {"x-goog-stored-content-length": "1048576", "Content-Length": "2097152"}
            ),
        ):
            size = _get_tokenizer_config_size("https://example.com/tokenizer.json")
        self.assertAlmostEqual(size, 1.0)

    def test_raises_value_error_when_no_size_header(self) -> None:
        """Chunked-transfer responses omit Content-Length; must raise ValueError, not TypeError."""
        with patch("requests.head", return_value=self._make_head_response({})):
            with self.assertRaises(ValueError) as ctx:
                _get_tokenizer_config_size("https://example.com/tokenizer.json")
        self.assertIn("Content-Length unavailable", str(ctx.exception))
