# This file was auto-generated by Fern from our API Definition.

import enum
import typing

T_Result = typing.TypeVar("T_Result")


class AuthTokenType(str, enum.Enum):
    """
    The token_type specifies the way the token is passed in the Authorization header. Valid values are "bearer", "basic", and "noscheme".
    """

    BEARER = "bearer"
    BASIC = "basic"
    NOSCHEME = "noscheme"

    def visit(
        self,
        bearer: typing.Callable[[], T_Result],
        basic: typing.Callable[[], T_Result],
        noscheme: typing.Callable[[], T_Result],
    ) -> T_Result:
        if self is AuthTokenType.BEARER:
            return bearer()
        if self is AuthTokenType.BASIC:
            return basic()
        if self is AuthTokenType.NOSCHEME:
            return noscheme()
