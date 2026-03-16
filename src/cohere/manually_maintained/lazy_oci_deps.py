"""Lazy loading for optional OCI SDK dependency."""

from typing import Any

OCI_INSTALLATION_MESSAGE = """
The OCI SDK is required to use OciClient or OciClientV2.

Install it with:
    pip install oci

Or with the optional dependency group:
    pip install cohere[oci]
"""


def lazy_oci() -> Any:
    """
    Lazily import the OCI SDK.

    Returns:
        The oci module

    Raises:
        ImportError: If the OCI SDK is not installed
    """
    try:
        import oci
        return oci
    except ImportError:
        raise ImportError(OCI_INSTALLATION_MESSAGE)
