import csv
import tempfile

from cohere.custom_model_dataset import CsvDataset, InMemoryDataset, JsonlDataset


def test_in_memory_dataset():
    dataset_without_eval = InMemoryDataset(training_data=[("prompt", "completion")])
    assert not dataset_without_eval.has_eval_file()
    decoded = csv.reader((row.decode("utf-8") for row in dataset_without_eval.get_train_data()))
    assert list(decoded) == [["prompt", "completion"]]

    dataset_with_eval = InMemoryDataset(
        training_data=[("prompt", "completion")], eval_data=[("eval_prompt", "eval_completion")]
    )
    assert dataset_with_eval.has_eval_file()
    train_decoded = csv.reader((row.decode("utf-8") for row in dataset_with_eval.get_train_data()))
    assert list(train_decoded) == [["prompt", "completion"]]
    eval_decoded = csv.reader((row.decode("utf-8") for row in dataset_with_eval.get_eval_data()))
    assert list(eval_decoded) == [["eval_prompt", "eval_completion"]]


def test_csv_dataset():
    with tempfile.NamedTemporaryFile("w", delete=True) as train_file:
        train_file.write("prompt,completion")
        train_file.seek(0)
        dataset = CsvDataset(train_file.name, delimiter=",")
        assert not dataset.has_eval_file()
        assert list(dataset.get_train_data()) == [b"prompt,completion"]

    with tempfile.NamedTemporaryFile("w", delete=True) as train_file:
        with tempfile.NamedTemporaryFile("w", delete=True) as eval_file:
            train_file.write("prompt,completion")
            eval_file.write("eval_prompt,eval_completion")
            train_file.seek(0)
            eval_file.seek(0)
            dataset = CsvDataset(train_file=train_file.name, eval_file=eval_file.name, delimiter=",")
            assert dataset.has_eval_file()
            assert list(dataset.get_train_data()) == [b"prompt,completion"]
            assert list(dataset.get_eval_data()) == [b"eval_prompt,eval_completion"]


def test_jsonl_dataset():
    with tempfile.NamedTemporaryFile("w", delete=True) as train_file:
        train_file.write('{"prompt": "prompt", "completion": "completion"}')
        train_file.seek(0)
        dataset = JsonlDataset(train_file.name)
        assert not dataset.has_eval_file()
        assert list(dataset.get_train_data()) == [b'{"prompt": "prompt", "completion": "completion"}']

    with tempfile.NamedTemporaryFile("w", delete=True) as train_file:
        with tempfile.NamedTemporaryFile("w", delete=True) as eval_file:
            train_file.write('{"prompt": "prompt", "completion": "completion"}')
            eval_file.write('{"prompt": "eval_prompt", "completion": "eval_completion"}')
            train_file.seek(0)
            eval_file.seek(0)
            dataset = JsonlDataset(train_file=train_file.name, eval_file=eval_file.name)
            assert dataset.has_eval_file()
            assert list(dataset.get_train_data()) == [b'{"prompt": "prompt", "completion": "completion"}']
            assert list(dataset.get_eval_data()) == [b'{"prompt": "eval_prompt", "completion": "eval_completion"}']


def test_text_dataset():
    with tempfile.NamedTemporaryFile("w", delete=True) as train_file:
        train_file.write("this is just a text")
        train_file.seek(0)
        dataset = JsonlDataset(train_file.name)
        assert not dataset.has_eval_file()
        assert list(dataset.get_train_data()) == [b"this is just a text"]

    with tempfile.NamedTemporaryFile("w", delete=True) as train_file:
        with tempfile.NamedTemporaryFile("w", delete=True) as eval_file:
            train_file.write("this is just a text")
            eval_file.write("this is just a eval text")
            train_file.seek(0)
            eval_file.seek(0)
            dataset = JsonlDataset(train_file=train_file.name, eval_file=eval_file.name)
            assert dataset.has_eval_file()
            assert list(dataset.get_train_data()) == [b"this is just a text"]
            assert list(dataset.get_eval_data()) == [b"this is just a eval text"]
