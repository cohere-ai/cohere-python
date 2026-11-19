import csv
import os
import tempfile
import typing
import unittest
from unittest import mock

from cohere.utils import save_csv

records: typing.List[typing.Dict[str, str]] = [
    {"text": "hello", "label": "a"},
    {"text": "goodbye", "label": "b"},
]


class TestSaveCsv(unittest.TestCase):

    def _write(self, path: str) -> bytes:
        with mock.patch("cohere.utils.dataset_generator", return_value=iter(records)):
            save_csv(mock.Mock(), path)
        with open(path, "rb") as f:
            return f.read()

    def test_save_csv_uses_a_single_crlf_per_row(self) -> None:
        # csv.DictWriter terminates rows with \r\n on every platform. If the
        # stream also translates newlines the file ends up with \r\r\n, which
        # readers see as a blank row between every record.
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._write(os.path.join(tmp, "out.csv"))

        self.assertEqual(raw, b"text,label\r\nhello,a\r\ngoodbye,b\r\n")

    def test_save_csv_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            self._write(path)
            with open(path, newline="") as f:
                self.assertEqual(list(csv.DictReader(f)), records)

    def test_save_csv_disables_newline_translation_on_the_stream(self) -> None:
        # The two tests above only fail on a platform whose text streams
        # translate newlines, so on Linux CI they pass either way. Assert the
        # open call directly as well, otherwise nothing here guards against the
        # newline argument being dropped again.
        with mock.patch("cohere.utils.dataset_generator", return_value=iter(records)), \
                mock.patch("cohere.utils.open", mock.mock_open(), create=True) as opened:
            save_csv(mock.Mock(), "out.csv")

        self.assertEqual(opened.call_args.kwargs.get("newline"), "")
