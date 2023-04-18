import csv
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Optional, Iterable, TypedDict


class FileConfig(TypedDict):
    separator: str
    switchColumns: bool
    hasHeader: bool
    delimiter: str


_empty_file_config = FileConfig(separator="", switchColumns=False, hasHeader=False, delimiter="")


class Dataset(ABC):
    @abstractmethod
    def file_config(self) -> FileConfig:
        pass

    @abstractmethod
    def train_file_name(self) -> str:
        ...

    @abstractmethod
    def eval_file_name(self) -> str:
        ...

    @abstractmethod
    def has_eval_file(self) -> str:
        ...

    @abstractmethod
    def get_train_file_contents(self) -> Iterable[bytes]:
        ...

    @abstractmethod
    def get_eval_file_contents(self) -> Iterable[bytes]:
        ...


class LocalFileDataset(Dataset):
    def __init__(self, train_file: str, eval_file: Optional[str] = None):
        self._train_file = train_file
        self._eval_file = eval_file

    @abstractmethod
    def file_config(self) -> FileConfig:
        pass

    def train_file_name(self) -> str:
        return Path(self._train_file).name

    def eval_file_name(self) -> str:
        if not self.has_eval_file():
            raise ValueError("Dataset has no eval file")
        return Path(self._eval_file).name

    def has_eval_file(self) -> str:
        return self._eval_file is not None

    def get_train_file_contents(self) -> Iterable[bytes]:
        with open(self._train_file, mode="rb") as _f:
            for line in _f:
                yield line

    def get_eval_file_contents(self) -> Iterable[bytes]:
        if not self.has_eval_file():
            raise ValueError("Dataset has no eval file")
        with open(self._eval_file, mode="rb") as _f:
            for line in _f:
                yield line


class CsvDataset(LocalFileDataset):
    def __init__(self, train_file: str, delimiter: str, eval_file: Optional[Path] = None):
        super().__init__(train_file, eval_file)
        self._delimiter = delimiter

    def file_config(self) -> FileConfig:
        config = dict(_empty_file_config)
        config["delimiter"] = self._delimiter
        return config


class JsonlDataset(LocalFileDataset):
    def file_config(self) -> FileConfig:
        return _empty_file_config.copy()


class TextDataset(LocalFileDataset):
    def __init__(self, train_file: str, separator: Optional[str], eval_file: Optional[Path] = None):
        super().__init__(train_file, eval_file)
        self._separator = separator

    def file_config(self) -> FileConfig:
        config = _empty_file_config.copy()
        if self._separator:
            config["separator"] = self._separator
        return config


class InMemoryDataset(Dataset):
    def __init__(self, training_data: Iterable[tuple[str, str]], eval_data: Optional[Iterable[tuple[str, str]]] = None):
        self._training_data = training_data
        self._eval_data = eval_data
        self._delimiter = ","

    def train_file_name(self) -> str:
        return "train.csv"

    def eval_file_name(self) -> str:
        if not self.has_eval_file():
            raise ValueError("Dataset has no eval data")
        return "eval.csv"

    def has_eval_file(self) -> str:
        return self._eval_data is not None

    def get_train_file_contents(self) -> Iterable[bytes]:
        for row in self._training_data:
            yield self._serialize_row(row)

    def get_eval_file_contents(self) -> Iterable[bytes]:
        if not self.has_eval_file():
            raise ValueError("Dataset has no eval data")
        for row in self._eval_data:
            yield self._serialize_row(row)

    def _serialize_row(self, row: tuple[str, str]) -> bytes:
        buffer = StringIO()
        writer = csv.writer(buffer, delimiter=self._delimiter)
        writer.writerow(row)
        return f"{buffer}\n".encode("utf-8")

    def file_config(self) -> FileConfig:
        config = _empty_file_config.copy()
        config["delimiter"] = self._delimiter
        return config
