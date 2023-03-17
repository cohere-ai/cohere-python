import json
import typing

from cohere.error import CohereError

try:  # numpy is optional, but support json encoding if the user has it
    import numpy as np

    class CohereJsonEncoder(json.JSONEncoder):
        """Handles numpy datatypes and such in json encoding"""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            else:
                return super().default(obj)

except:

    class CohereJsonEncoder(json.JSONEncoder):
        """numpy is missing, so we can't handle these (and don't expect them)"""

        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            else:
                return super().default(obj)


def np_json_dumps(data, **kwargs):
    return json.dumps(data, cls=CohereJsonEncoder, **kwargs)


def is_api_key_valid(key: typing.Optional[str]) -> bool:
    """is_api_key_valid returns True when the key is valid and raises a CohereError when it is invalid."""
    if not key:
        raise CohereError(
            "No API key provided. Provide the API key in the client initialization or the CO_API_KEY environment variable."  # noqa: E501
        )

    return True
