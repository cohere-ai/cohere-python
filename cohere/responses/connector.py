from typing import Any, Dict

from cohere.responses.base import CohereObject
from cohere.utils import parse_datetime


class ConnectorOAuth(CohereObject):
    def __init__(
        self,
        authorize_url: str,
        token_url: str,
        scope: str,
    ) -> None:
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.scope = scope

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectorOAuth":
        return cls(authorize_url=data["authorize_url"], token_url=data["token_url"], scope=data.get("scope", ""))


class Connector(CohereObject):
    def __init__(
        self,
        id: str,
        organization_id: str,
        name: str,
        url: str,
        created_at: str,
        updated_at: str,
        auth_type: str,
        active: bool,
        continue_on_failure: bool,
        oauth: ConnectorOAuth | None,
        auth_status: str,
    ) -> None:
        self.id = id
        self.organization_id = organization_id
        self.name = name
        self.url = url
        self.created_at = parse_datetime(created_at)
        self.updated_at = parse_datetime(updated_at)
        self.auth_type = auth_type
        self.active = active
        self.continue_on_failure = continue_on_failure
        self.oauth = oauth
        self.auth_status = auth_status

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Connector":
        return cls(
            id=data["id"],
            name=data["name"],
            url=data.get("url", ""),
            organization_id=data.get("organization_id", ""),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            auth_type=data.get("auth_type", ""),
            active=data.get("active", True),
            continue_on_failure=data["continue_on_failure"],
            oauth=ConnectorOAuth.from_dict(data["oauth"]) if "oauth" in data else None,
            auth_status=data.get("auth_status", ""),
        )
