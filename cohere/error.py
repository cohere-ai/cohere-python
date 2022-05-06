class CohereError(Exception):
    def __init__(
        self,
        message=None,
        http_status=None,
        headers=None,
    ) -> None:
        super(CohereError, self).__init__(message)

        self.message = message
        self.http_status = http_status
        self.headers = headers or {}

    def __str__(self) -> str:
        msg = self.message or '<empty message>'
        return msg

    def __repr__(self) -> str:
        return '%s(message=%r, http_status=%r)' % (
            self.__class__.__name__,
            self.message,
            self.http_status,
        )
