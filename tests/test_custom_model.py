from datetime import datetime, timezone

from cohere.responses.custom_model import ModelMetric, _parse_date_with_variable_seconds


def test_custom_model_from_dict_with_all_fields_set():
    as_dict = {
        "accuracy": 0.08035714,
        "created_at": "2023-06-28T16:37:22.31835556Z",
        "f1": 0.029752064,
        "loss": 0.19185612,
        "precision": 0.016071428,
        "recall": 0.2,
        "step_num": 0,
    }
    actual = ModelMetric.from_dict(as_dict)
    expected = ModelMetric(
        created_at=datetime(2023, 6, 28, 16, 37, 22, 318355, tzinfo=timezone.utc),
        step_num=0,
        accuracy=0.08035714,
        f1=0.029752064,
        loss=0.19185612,
        precision=0.016071428,
        recall=0.2,
    )
    assert actual == expected


def test_custom_model_from_dict_with_metrics_missing():
    as_dict = {
        "created_at": "2023-06-28T16:37:22.31835556Z",
        "step_num": 0,
    }
    actual = ModelMetric.from_dict(as_dict)
    expected = ModelMetric(
        created_at=datetime(2023, 6, 28, 16, 37, 22, 318355, tzinfo=timezone.utc),
        step_num=0,
    )
    assert actual == expected


def test_parse_date_with_variable_seconds():
    nanosec_time = "2023-06-28T16:37:22.318355568Z"
    actual_ns = _parse_date_with_variable_seconds(nanosec_time)
    expected_ns = datetime(2023, 6, 28, 16, 37, 22, 318355, tzinfo=timezone.utc)
    assert actual_ns == expected_ns

    normal_time = "2023-06-28T16:37:22.31835556Z"
    actual_s = _parse_date_with_variable_seconds(normal_time)
    expected_s = datetime(2023, 6, 28, 16, 37, 22, 318355, tzinfo=timezone.utc)
    assert actual_s == expected_s
