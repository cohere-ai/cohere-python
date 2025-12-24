import unittest
from unittest.mock import Mock
from json.decoder import JSONDecodeError

from cohere.v2.raw_client import _safe_json_parse


class TestSafeJsonParse(unittest.TestCase):
    """Test the _safe_json_parse helper function"""

    def test_valid_json_response(self) -> None:
        """Test that valid JSON is parsed correctly"""
        mock_response = Mock()
        mock_response.json.return_value = {"key": "value", "status": "success"}

        result = _safe_json_parse(mock_response)

        self.assertEqual(result, {"key": "value", "status": "success"})
        mock_response.json.assert_called_once()

    def test_empty_response_body(self) -> None:
        """Test that empty response body returns text instead of raising JSONDecodeError"""
        mock_response = Mock()
        mock_response.json.side_effect = JSONDecodeError("Expecting value", "", 0)
        mock_response.text = ""

        result = _safe_json_parse(mock_response)

        self.assertEqual(result, "")
        mock_response.json.assert_called_once()

    def test_malformed_json_response(self) -> None:
        """Test that malformed JSON returns text instead of raising JSONDecodeError"""
        mock_response = Mock()
        mock_response.json.side_effect = JSONDecodeError("Expecting value", "not json", 0)
        mock_response.text = "Internal Server Error"

        result = _safe_json_parse(mock_response)

        self.assertEqual(result, "Internal Server Error")
        mock_response.json.assert_called_once()

    def test_500_error_with_empty_body(self) -> None:
        """Test the actual production error case: HTTP 500 with empty response body"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = JSONDecodeError("Expecting value: line 1 column 1 (char 0)", "", 0)
        mock_response.text = ""

        result = _safe_json_parse(mock_response)

        self.assertEqual(result, "")
        self.assertIsInstance(result, str)
        mock_response.json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
