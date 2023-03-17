from typing import Dict


class CohereError(Exception):
    """Base Exception class, returned when nothing more specific applies"""

    def __init__(self, message=None) -> None:
        super(CohereError, self).__init__(message)

        self.message = message

    def __str__(self) -> str:
        msg = self.message or "<empty message>"
        return msg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={str(self)})"


class CohereAPIError(CohereError):
    """Returned when the API responds with an error message"""

    def __init__(self, message: str = None, http_status: int = None, headers: Dict = None):
        super().__init__(message)
        self.http_status = http_status
        self.headers = headers or {}

    @classmethod
    def from_response(cls, response, message=None):
        return cls(message=message or response.text, http_status=response.status, headers=response.headers)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={str(self)}, http_status={self.http_status})"


class CohereConnectionError(CohereError):
    """Returned when the SDK can not reach the API server for any reason"""
