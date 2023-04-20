from typing import Dict

from cohere.error import CohereAPIError, CohereError
from cohere.logging import logger


def check_response(json_response: Dict, headers: Dict, status_code: int):
    if "X-API-Warning" in headers:
        logger.warning(headers["X-API-Warning"])
    if "message" in json_response:  # has errors
        raise CohereAPIError(
            message=json_response["message"],
            http_status=status_code,
            headers=headers,
        )
    if 400 <= status_code < 500:
        raise CohereAPIError(
            message=f"Unexpected client error (status {status_code}): {json_response}",
            http_status=status_code,
            headers=headers,
        )
    if status_code >= 500:
        raise CohereError(message=f"Unexpected server error (status {status_code}): {json_response}")
