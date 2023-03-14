from typing import NamedTuple

APIVersion = NamedTuple("api_version", [("version", str), ("is_deprecated", bool), ("is_experimental", bool)])
Meta = NamedTuple("meta", [("api_version", APIVersion)])
