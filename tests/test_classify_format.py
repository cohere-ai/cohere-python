from cohere import Client


def test_classifcation_old_single_label_format(monkeypatch):
    response = {
        "id": "8a2c7187-6c01-41c0-a241-c064ad9618a5",
        "classifications": [
            {
                "classification_type": "single-label",
                "confidence": 0.24627389,
                "confidences": [0.24627389],
                "id": "d0dfe4ce-525d-4530-ab26-ded93a101116",
                "input": "I don't like this movie",
                "labels": {
                    "negative": {"confidence": 0.24627389},
                    "neutral": {"confidence": 0.18561405},
                    "positive": {"confidence": 0.1925146},
                    "very negative": {"confidence": 0.20908539},
                    "very positive": {"confidence": 0.16651207},
                },
                "prediction": "negative",
                "predictions": ["negative"],
            },
        ],
        "meta": {"api_version": {"version": "1"}},
    }
    monkeypatch.setattr("cohere.Client._request", lambda *args, **kwargs: response)
    co = Client("test_token")
    result = co.classify(["i don't like this movie"], model="sentence classifier single label old")
    # Both deprecated fields (prediction/confidence) and new fields (predictions/confidences) are supported
    assert result[0].predictions == ["negative"]
    assert result[0].confidences == [0.24627389]
    assert result[0].prediction == "negative"
    assert result[0].confidence == 0.24627389
    assert not result[0].is_multilabel()


def test_classify_new_single_label_format(monkeypatch):
    response = {
        "id": "e994e80f-08b1-402f-8653-ced25a946f3a",
        "classifications": [
            {
                "classification_type": "single-label",
                "confidence": 0.8908454,
                "confidences": [0.8908454],
                "id": "b9823024-3ad1-47d5-aed9-2bc4cb7775c8",
                "input": "i love this movie!",
                "labels": {
                    "negative": {"confidence": 7.224075e-05},
                    "neutral": {"confidence": 0.0011411251},
                    "positive": {"confidence": 0.10786094},
                    "very negative": {"confidence": 8.027619e-05},
                    "very positive": {"confidence": 0.8908454},
                },
                "prediction": "very positive",
                "predictions": ["very positive"],
            },
        ],
        "meta": {"api_version": {"version": "1"}},
    }
    monkeypatch.setattr("cohere.Client._request", lambda *args, **kwargs: response)
    co = Client("test_token")
    result = co.classify(["i love this movie!"], model="sentence classifier single label new")
    # Both deprecated fields (prediction/confidence) and new fields (predictions/confidences) are supported
    assert result[0].predictions == ["very positive"]
    assert result[0].confidences == [0.8908454]
    assert result[0].prediction == "very positive"
    assert result[0].confidence == 0.8908454
    assert not result[0].is_multilabel()


def test_classify_multilabel_format(monkeypatch):
    response = {
        "id": "cee2e2c2-83be-4c99-ad46-288448000b3f",
        "classifications": [
            {
                "classification_type": "multi-label",
                "confidences": [0.6740505],
                "id": "ff5b50c5-3f07-4993-9345-d47d71736164",
                "input": "i love this movie!",
                "labels": {
                    "0": {"confidence": 0.005260852},
                    "1": {"confidence": 0.0029810327},
                    "2": {"confidence": 0.000119598575},
                    "3": {"confidence": 5.507606e-06},
                    "4": {"confidence": 0.00055277866},
                    "5": {"confidence": 0.00054847926},
                    "6": {"confidence": 0.6740505},
                    "7": {"confidence": 0.017242778},
                    "8": {"confidence": 0.00026323833},
                    "9": {"confidence": 0.00012533751},
                },
                "predictions": ["6"],
            },
        ],
        "meta": {"api_version": {"version": "1"}},
    }
    monkeypatch.setattr("cohere.Client._request", lambda *args, **kwargs: response)
    co = Client("test_token")
    result = co.classify(["i love this movie!"], model="sentence classifier multi label new")
    # prediction/confidence do not make sense for multi-label classification
    assert result[0].predictions == ["6"]
    assert result[0].confidences == [0.6740505]
    assert result[0].is_multilabel()
