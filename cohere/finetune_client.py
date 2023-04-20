import json as jsonlib
import os
from datetime import datetime, timezone
from typing import Any, Iterable, List, Literal, Optional, TypedDict

import requests

import cohere
from cohere.error import CohereAPIError, CohereConnectionError, CohereError
from cohere.finetune_dataset import Dataset
from cohere.responses.finetune import (
    FINETUNE_PRODUCT_MAPPING,
    FINETUNE_STATUS,
    FINETUNE_TYPE,
    INTERNAL_FINETUNE_TYPE,
    Finetune,
)
from cohere.responses.response import check_response


class FinetuneClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = cohere.COHERE_API_URL
        self._request_source = "python-sdk"

    def get(self, finetune_id: str) -> Finetune:
        """Get a finetune by id.

        Args:
            finetune_id (str): finetune id
        Returns:
            Finetune: the finetune
        """
        json = {"finetuneID": finetune_id}
        response = self._request(f"{cohere.BLOBHEART_URL}/GetFinetune", method="POST", json=json)
        return Finetune.from_dict(response["finetune"])

    def get_by_name(self, name: str) -> Finetune:
        """Get a finetune by name.

        Args:
            name (str): finetune name
        Returns:
            Finetune: the finetune
        """
        json = {"name": name}
        response = self._request(f"{cohere.BLOBHEART_URL}/GetFinetuneByName", method="POST", json=json)
        return Finetune.from_dict(response["finetune"])

    def list(
        self,
        statuses: Optional[List[FINETUNE_STATUS]] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        order_by: Optional[Literal["asc", "desc"]] = None,
    ) -> List[Finetune]:
        """List finetunes of your organization.

        Args:
            statuses (FINETUNE_STATUS, optional): search for fintunes which are in one of these states
            before (datetime, optional): search for finetunes that were created before this timestamp
            after (datetime, optional): search for finetunes that were created after this timestamp
            order_by (Literal["asc", "desc"], optional): sort finetunes by created at, either asc or desc
        Returns:
            List[Finetune]: a list of finetunes.
        """
        if before:
            before = before.replace(tzinfo=before.tzinfo or timezone.utc)
        if after:
            after = after.replace(tzinfo=after.tzinfo or timezone.utc)

        json = {
            "query": {
                "statuses": statuses,
                "before": before.isoformat(timespec="seconds") if before else None,
                "after": after.isoformat(timespec="seconds") if after else None,
                "orderBy": order_by,
            }
        }

        response = self._request(f"{cohere.BLOBHEART_URL}/ListFinetunes", method="POST", json=json)
        return [Finetune.from_dict(r) for r in response["finetunes"]]

    def create(self, name: str, finetune_type: FINETUNE_TYPE, dataset: Dataset) -> str:
        """Create a new finetune

        Args:
            name (str): name of your finetune, has to be unique across your organization
            finetune_type (GENERATIVE, CONTRASTIVE, CLASSIFICATION): type of finetune
            dataset (InMemoryDataset, CsvDataset, JsonlDataset, TextDataset): A dataset for your training. Consists of a train and optional eval file.
        Returns:
            str: the id of the finetune that was created

        Examples:
            prompt completion finetune with csv file
            >>> from cohere.finetune_dataset import CsvDataset
            >>> co = cohere.Client("YOUR_API_KEY")
            >>> dataset = CsvDataset(train_file="/path/to/your/file.csv", delimiter=",")
            >>> finetune = co.finetune.create("prompt-completion-ft", dataset=dataset, finetune_type="GENERATIVE")

            prompt completion finetune with in-memory dataset
            >>> from cohere.finetune_dataset import InMemoryDataset
            >>> co = cohere.Client("YOUR_API_KEY")
            >>> dataset = InMemoryDataset(training_data=[
            >>>     ("this is the prompt", "and this is the completion"),
            >>>     ("another prompt", "and another completion")
            >>> ])
            >>> finetune = co.finetune.create("prompt-completion-ft", dataset=dataset, finetune_type="GENERATIVE")

        """
        internal_finetune_type = FINETUNE_PRODUCT_MAPPING[finetune_type]
        json = {
            "name": name,
            "settings": {
                "trainFiles": [],
                "evalFiles": [],
                "baseModel": "medium",
                "finetuneType": internal_finetune_type,
            },
        }
        remote_path = self._upload_dataset(
            dataset.get_train_data(), name, dataset.train_file_name(), internal_finetune_type
        )
        json["settings"]["trainFiles"].append({"path": remote_path, **dataset.file_config()})
        if dataset.has_eval_file():
            remote_path = self._upload_dataset(
                dataset.get_eval_data(), name, dataset.eval_file_name(), internal_finetune_type
            )
            json["settings"]["evalFiles"].append({"path": remote_path, **dataset.file_config()})

        response = self._request(f"{cohere.BLOBHEART_URL}/CreateFinetune", method="POST", json=json)
        return response["finetune"]["id"]

    def _upload_dataset(
        self, content: Iterable[bytes], finetune_name: str, file_name: str, type: INTERNAL_FINETUNE_TYPE
    ) -> str:
        gcs = self._create_signed_url(finetune_name, file_name, type)
        response = requests.put(gcs["url"], data=content, headers={"content-type": "text/plain"})
        check_response({}, dict(response.headers), response.status_code)
        return gcs["gcspath"]

    def _create_signed_url(
        self, finetune_name: str, file_name: str, type: INTERNAL_FINETUNE_TYPE
    ) -> TypedDict("gcsData", {"url": str, "gcspath": str}):
        json = {"finetuneName": finetune_name, "fileName": file_name, "finetuneType": type}
        return self._request(f"{cohere.BLOBHEART_URL}/GetFinetuneUploadSignedURL", method="POST", json=json)

    def _request(self, endpoint, json=None, method="POST", headers: Optional[dict[str, str]] = None) -> Any:
        if not headers:
            headers = {}

        headers.update(
            {
                "Authorization": "BEARER {}".format(self._api_key),
                "Content-Type": "application/json",
                "Request-Source": self._request_source,
            }
        )

        url = f"{self._api_url}/{endpoint}"

        try:
            response = requests.request(method, url, headers=headers, json=json)
        except requests.exceptions.ConnectionError as e:
            raise CohereConnectionError(str(e)) from e
        except requests.exceptions.RequestException as e:
            raise CohereError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

        try:
            json_response = response.json()
        except jsonlib.decoder.JSONDecodeError:  # CohereAPIError will capture status
            raise CohereAPIError.from_response(response, message=f"Failed to decode json body: {response.text}")

        check_response(json_response, dict(response.headers), response.status_code)
        return json_response
