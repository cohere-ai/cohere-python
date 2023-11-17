from datetime import datetime, timezone

from cohere.responses.custom_model import (
    ModelMetric,
    _parse_date_with_variable_seconds,
    FinetuneBilling,
    CustomModel,
    HyperParameters,
)


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


def test_finetune_billing():
    as_dict = {
        "numTrainingTokens": 4506,
        "epochs": 1,
        "unitPrice": 0.000001,
        "totalCost": 0.004506,
    }
    actual = FinetuneBilling.from_response(as_dict)
    expected = FinetuneBilling(train_epochs=1, num_training_tokens=4506, unit_price=0.000001, total_cost=0.004506)
    assert actual == expected


def test_finetune_response_with_billing():
    response_dict = {
        "finetune": {
            "id": "dd942318-dac4-44c2-866e-ce95396e3b00",
            "name": "test-response-2",
            "creator_id": "91d102b7-b2b9-464a-aa5e-85569a49aa6d",
            "organization_id": "7e17242a-8489-4650-9631-f9bcb49319bd",
            "organization_name": "",
            "status": "QUEUED",
            "created_at": "2023-11-17T17:24:36.769824Z",
            "updated_at": "2023-11-17T17:24:36.769824Z",
            "settings": {
                "datasetID": "",
                "trainFiles": [
                    {
                        "path": "gs://cohere-dev/blobheart-uploads/staging/91d102b7-b2b9-464a-aa5e-85569a49aa6d/wnvz7i/GENERATIVE/train.csv",
                        "separator": "",
                        "switchColumns": False,
                        "hasHeader": False,
                        "delimiter": ",",
                    }
                ],
                "evalFiles": [],
                "baseModel": "medium",
                "finetuneType": "GENERATIVE",
                "faxOverride": None,
                "finetuneStrategy": "TFEW",
                "hyperparameters": {
                    "earlyStoppingPatience": 6,
                    "earlyStoppingThreshold": 0.01,
                    "trainBatchSize": 16,
                    "trainSteps": 2,
                    "trainEpochs": 1,
                    "learningRate": 0.01,
                },
                "baseVersion": "14.2.0",
                "multiLabel": False,
            },
            "model": {
                "name": "mike-test-response-2",
                "route": "dd942318-dac4-44c2-866e-ce95396e3b00-ft",
                "endpoints": ["generate"],
                "isFinetune": True,
                "isProtected": False,
                "languages": None,
            },
            "data_metrics": {
                "train_files": [{"name": "train.csv", "totalExamples": 32, "size": 140}],
                "total_examples": 32,
                "trainable_token_count": 192,
            },
            "billing": {"numTrainingTokens": 192, "epochs": 1, "unitPrice": 0.000001, "totalCost": 0.000192},
        }
    }
    actual = CustomModel.from_dict(response_dict["finetune"], None)
    expect = CustomModel(
        id="dd942318-dac4-44c2-866e-ce95396e3b00",
        name="test-response-2",
        status="QUEUED",
        created_at=datetime(2023, 11, 17, 17, 24, 36, 769824, tzinfo=timezone.utc),
        model_id="dd942318-dac4-44c2-866e-ce95396e3b00-ft",
        base_model="medium",
        hyperparameters=HyperParameters(
            early_stopping_patience=6,
            early_stopping_threshold=0.01,
            train_batch_size=16,
            train_steps=2,
            train_epochs=1,
            learning_rate=0.01,
        ),
        model_type="GENERATIVE",
        billing=FinetuneBilling(train_epochs=1, num_training_tokens=192, unit_price=0.000001, total_cost=0.000192),
        completed_at=None,
        wait_fn=None,
    )
    assert actual.__dict__ == expect.__dict__
