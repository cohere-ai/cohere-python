# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from .classify_response_classifications_item_classification_type import (
    ClassifyResponseClassificationsItemClassificationType,
)
from .classify_response_classifications_item_labels_value import ClassifyResponseClassificationsItemLabelsValue

try:
    import pydantic.v1 as pydantic  # type: ignore
except ImportError:
    import pydantic  # type: ignore


class ClassifyResponseClassificationsItem(pydantic.BaseModel):
    id: str
    input: typing.Optional[str] = pydantic.Field(description="The input text that was classified")
    prediction: typing.Optional[str] = pydantic.Field(
        description="The predicted label for the associated query (only filled for single-label models)"
    )
    predictions: typing.List[str] = pydantic.Field(
        description="An array containing the predicted labels for the associated query (only filled for single-label classification)"
    )
    confidence: typing.Optional[float] = pydantic.Field(
        description="The confidence score for the top predicted class (only filled for single-label classification)"
    )
    confidences: typing.List[float] = pydantic.Field(
        description="An array containing the confidence scores of all the predictions in the same order"
    )
    labels: typing.Dict[str, ClassifyResponseClassificationsItemLabelsValue] = pydantic.Field(
        description="A map containing each label and its confidence score according to the classifier. All the confidence scores add up to 1 for single-label classification. For multi-label classification the label confidences are independent of each other, so they don't have to sum up to 1."
    )
    classification_type: ClassifyResponseClassificationsItemClassificationType = pydantic.Field(
        description="The type of classification performed"
    )

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().dict(**kwargs_with_defaults)

    class Config:
        frozen = True
        smart_union = True
        json_encoders = {dt.datetime: serialize_datetime}
