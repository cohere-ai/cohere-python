# This file was auto-generated by Fern from our API Definition.

import typing

Status = typing.Union[
    typing.AnyStr,
    typing.Literal[
        "STATUS_UNSPECIFIED",
        "STATUS_FINETUNING",
        "STATUS_DEPLOYING_API",
        "STATUS_READY",
        "STATUS_FAILED",
        "STATUS_DELETED",
        "STATUS_TEMPORARILY_OFFLINE",
        "STATUS_PAUSED",
        "STATUS_QUEUED",
    ],
]
