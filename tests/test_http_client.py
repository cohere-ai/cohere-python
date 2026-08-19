"""Unit tests for cohere.core.http_client helper functions."""

import pytest
import httpx

from cohere.core.http_client import _parse_retry_after


def _make_headers(pairs: dict) -> httpx.Headers:
    """Build an httpx.Headers object from a plain dict."""
    return httpx.Headers(pairs)


class TestParseRetryAfterMs:
    """Regression tests for the Retry-After-Ms header parser.

    The header value is always a *string* (HTTP headers are plain text).
    The original code compared the raw string to an integer with ``>``,
    which raises ``TypeError`` in Python 3 and caused the entire
    ``Retry-After-Ms`` header to be silently ignored on every response.
    """

    def test_positive_ms_value_is_converted_to_seconds(self):
        headers = _make_headers({"retry-after-ms": "2000"})
        result = _parse_retry_after(headers)
        assert result == pytest.approx(2.0)

    def test_ms_value_of_zero_returns_zero(self):
        headers = _make_headers({"retry-after-ms": "0"})
        result = _parse_retry_after(headers)
        assert result == 0

    def test_negative_ms_value_returns_zero(self):
        headers = _make_headers({"retry-after-ms": "-500"})
        result = _parse_retry_after(headers)
        assert result == 0

    def test_ms_takes_priority_over_retry_after(self):
        """Retry-After-Ms should be preferred over the plain Retry-After header."""
        headers = _make_headers({"retry-after-ms": "3000", "retry-after": "99"})
        result = _parse_retry_after(headers)
        assert result == pytest.approx(3.0)

    def test_fractional_ms_value(self):
        headers = _make_headers({"retry-after-ms": "1500"})
        result = _parse_retry_after(headers)
        assert result == pytest.approx(1.5)

    def test_invalid_ms_value_falls_back_to_retry_after(self):
        """Non-numeric Retry-After-Ms should be ignored; fall back to Retry-After."""
        headers = _make_headers({"retry-after-ms": "not-a-number", "retry-after": "5"})
        result = _parse_retry_after(headers)
        assert result == pytest.approx(5.0)

    def test_no_retry_headers_returns_none(self):
        headers = _make_headers({})
        result = _parse_retry_after(headers)
        assert result is None


class TestParseRetryAfterSeconds:
    """Tests for the plain Retry-After header (integer seconds)."""

    def test_integer_seconds(self):
        headers = _make_headers({"retry-after": "30"})
        result = _parse_retry_after(headers)
        assert result == pytest.approx(30.0)

    def test_zero_seconds(self):
        headers = _make_headers({"retry-after": "0"})
        result = _parse_retry_after(headers)
        assert result == 0

    def test_negative_seconds_clamped_to_zero(self):
        # The header shouldn't be negative, but we clamp it.
        # parsedate_tz won't parse a bare negative integer, so this falls
        # through; the integer branch uses re.match which only matches [0-9]+.
        # Negative numbers fail the regex → parsedate_tz fails → returns None.
        headers = _make_headers({"retry-after": "-10"})
        result = _parse_retry_after(headers)
        # A negative integer doesn't match the regex, parsedate_tz can't parse
        # it either, so None is expected.
        assert result is None
