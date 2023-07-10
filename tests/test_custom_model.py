from datetime import datetime, timezone

from cohere.responses.custom_model import ModelMetric


def test_custom_model_from_dict():
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
        accuracy=0.08035714,
        created_at=datetime(2023, 6, 28, 16, 37, 22, 318355, tzinfo=timezone.utc),
        f1=0.029752064,
        loss=0.19185612,
        precision=0.016071428,
        recall=0.2,
        step_num=0,
    )
    assert actual == expected
