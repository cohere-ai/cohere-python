"""Oracle Cloud Infrastructure (OCI) client for Cohere API."""

import configparser
import email.utils
import json
import os
import typing
import uuid

import httpx
import requests
from .client import Client, ClientEnvironment
from .client_v2 import ClientV2
from .aws_client import Streamer
from .manually_maintained.lazy_oci_deps import lazy_oci
from httpx import URL, ByteStream


class OciClient(Client):
    """
    Cohere V1 API client for Oracle Cloud Infrastructure (OCI) Generative AI service.

    Use this client for V1 API models (Command R family) and embeddings.
    For V2 API models (Command A family), use OciClientV2 instead.

    Supported APIs on OCI:
    - embed(): Full support for all embedding models
    - chat(): Full support with Command-R models
    - chat_stream(): Streaming chat support

    Supports all authentication methods:
    - Config file (default): Uses ~/.oci/config
    - Session-based: Uses OCI CLI session tokens
    - Direct credentials: Pass OCI credentials directly
    - Instance principal: For OCI compute instances
    - Resource principal: For OCI functions

    Example:
        ```python
        import cohere

        client = cohere.OciClient(
            oci_region="us-chicago-1",
            oci_compartment_id="ocid1.compartment.oc1...",
        )

        response = client.chat(
            model="command-r-08-2024",
            message="Hello!",
        )
        print(response.text)
        ```
    """

    def __init__(
        self,
        *,
        oci_config_path: typing.Optional[str] = None,
        oci_profile: typing.Optional[str] = None,
        oci_user_id: typing.Optional[str] = None,
        oci_fingerprint: typing.Optional[str] = None,
        oci_tenancy_id: typing.Optional[str] = None,
        oci_private_key_path: typing.Optional[str] = None,
        oci_private_key_content: typing.Optional[str] = None,
        auth_type: typing.Literal["api_key", "instance_principal", "resource_principal"] = "api_key",
        oci_region: typing.Optional[str] = None,
        oci_compartment_id: str,
        timeout: typing.Optional[float] = None,
    ):
        oci_config = _load_oci_config(
            auth_type=auth_type,
            config_path=oci_config_path,
            profile=oci_profile,
            user_id=oci_user_id,
            fingerprint=oci_fingerprint,
            tenancy_id=oci_tenancy_id,
            private_key_path=oci_private_key_path,
            private_key_content=oci_private_key_content,
        )

        if oci_region is None:
            oci_region = oci_config.get("region")
            if oci_region is None:
                raise ValueError("oci_region must be provided either directly or in OCI config file")

        Client.__init__(
            self,
            base_url="https://api.cohere.com",
            environment=ClientEnvironment.PRODUCTION,
            client_name="n/a",
            timeout=timeout,
            api_key="n/a",
            httpx_client=httpx.Client(
                event_hooks=get_event_hooks(
                    oci_config=oci_config,
                    oci_region=oci_region,
                    oci_compartment_id=oci_compartment_id,
                    is_v2_client=False,
                ),
                timeout=timeout,
            ),
        )


class OciClientV2(ClientV2):
    """
    Cohere V2 API client for Oracle Cloud Infrastructure (OCI) Generative AI service.

    Supported APIs on OCI:
    - embed(): Full support for all embedding models (returns embeddings as dict)
    - chat(): Full support with Command-A models (command-a-03-2025)
    - chat_stream(): Streaming chat with proper V2 event format

    Note: rerank() requires fine-tuned models deployed to dedicated endpoints.
    OCI on-demand inference does not support the rerank API.

    Supports all authentication methods:
    - Config file (default): Uses ~/.oci/config
    - Session-based: Uses OCI CLI session tokens
    - Direct credentials: Pass OCI credentials directly
    - Instance principal: For OCI compute instances
    - Resource principal: For OCI functions

    Example using config file:
        ```python
        import cohere

        client = cohere.OciClientV2(
            oci_region="us-chicago-1",
            oci_compartment_id="ocid1.compartment.oc1...",
        )

        response = client.embed(
            model="embed-english-v3.0",
            texts=["Hello world"],
            input_type="search_document",
        )
        print(response.embeddings.float_)

        response = client.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.message)
        ```

    Example using direct credentials:
        ```python
        client = cohere.OciClientV2(
            oci_user_id="ocid1.user.oc1...",
            oci_fingerprint="xx:xx:xx:...",
            oci_tenancy_id="ocid1.tenancy.oc1...",
            oci_private_key_path="~/.oci/key.pem",
            oci_region="us-chicago-1",
            oci_compartment_id="ocid1.compartment.oc1...",
        )
        ```

    Example using instance principal:
        ```python
        client = cohere.OciClientV2(
            auth_type="instance_principal",
            oci_region="us-chicago-1",
            oci_compartment_id="ocid1.compartment.oc1...",
        )
        ```
    """

    def __init__(
        self,
        *,
        # Authentication - Config file (default)
        oci_config_path: typing.Optional[str] = None,
        oci_profile: typing.Optional[str] = None,
        # Authentication - Direct credentials
        oci_user_id: typing.Optional[str] = None,
        oci_fingerprint: typing.Optional[str] = None,
        oci_tenancy_id: typing.Optional[str] = None,
        oci_private_key_path: typing.Optional[str] = None,
        oci_private_key_content: typing.Optional[str] = None,
        # Authentication - Instance principal
        auth_type: typing.Literal["api_key", "instance_principal", "resource_principal"] = "api_key",
        # Required for OCI Generative AI
        oci_region: typing.Optional[str] = None,
        oci_compartment_id: str,
        # Standard parameters
        timeout: typing.Optional[float] = None,
    ):
        # Load OCI config based on auth_type
        oci_config = _load_oci_config(
            auth_type=auth_type,
            config_path=oci_config_path,
            profile=oci_profile,
            user_id=oci_user_id,
            fingerprint=oci_fingerprint,
            tenancy_id=oci_tenancy_id,
            private_key_path=oci_private_key_path,
            private_key_content=oci_private_key_content,
        )

        # Get region from config if not provided
        if oci_region is None:
            oci_region = oci_config.get("region")
            if oci_region is None:
                raise ValueError("oci_region must be provided either directly or in OCI config file")

        # Create httpx client with OCI event hooks
        ClientV2.__init__(
            self,
            base_url="https://api.cohere.com",  # Unused, OCI URL set in hooks
            environment=ClientEnvironment.PRODUCTION,
            client_name="n/a",
            timeout=timeout,
            api_key="n/a",
            httpx_client=httpx.Client(
                event_hooks=get_event_hooks(
                    oci_config=oci_config,
                    oci_region=oci_region,
                    oci_compartment_id=oci_compartment_id,
                    is_v2_client=True,
                ),
                timeout=timeout,
            ),
        )


EventHook = typing.Callable[..., typing.Any]


def _load_oci_config(
    auth_type: str,
    config_path: typing.Optional[str],
    profile: typing.Optional[str],
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    """
    Load OCI configuration based on authentication type.

    Args:
        auth_type: Authentication method (api_key, instance_principal, resource_principal)
        config_path: Path to OCI config file (for api_key auth)
        profile: Profile name in config file (for api_key auth)
        **kwargs: Direct credentials (user_id, fingerprint, etc.)

    Returns:
        Dictionary containing OCI configuration
    """
    oci = lazy_oci()

    if auth_type == "instance_principal":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return {"signer": signer, "auth_type": "instance_principal"}

    elif auth_type == "resource_principal":
        signer = oci.auth.signers.get_resource_principals_signer()
        return {"signer": signer, "auth_type": "resource_principal"}

    elif kwargs.get("user_id"):
        # Direct credentials provided - validate required fields
        required_fields = ["fingerprint", "tenancy_id"]
        missing = [f for f in required_fields if not kwargs.get(f)]
        if missing:
            raise ValueError(
                f"When providing oci_user_id, you must also provide: {', '.join('oci_' + f for f in missing)}"
            )
        if not kwargs.get("private_key_path") and not kwargs.get("private_key_content"):
            raise ValueError(
                "When providing oci_user_id, you must also provide either "
                "oci_private_key_path or oci_private_key_content"
            )
        config = {
            "user": kwargs["user_id"],
            "fingerprint": kwargs["fingerprint"],
            "tenancy": kwargs["tenancy_id"],
        }
        if kwargs.get("private_key_path"):
            config["key_file"] = kwargs["private_key_path"]
        if kwargs.get("private_key_content"):
            config["key_content"] = kwargs["private_key_content"]
        return config

    else:
        # Load from config file
        oci_config = oci.config.from_file(
            file_location=config_path or "~/.oci/config", profile_name=profile or "DEFAULT"
        )
        _remove_inherited_session_auth(oci_config, config_path=config_path, profile=profile)
        return oci_config


def _remove_inherited_session_auth(
    oci_config: typing.Dict[str, typing.Any],
    *,
    config_path: typing.Optional[str],
    profile: typing.Optional[str],
) -> None:
    """Drop session auth fields inherited from the OCI config DEFAULT section."""
    profile_name = profile or "DEFAULT"
    if profile_name == "DEFAULT" or "security_token_file" not in oci_config:
        return

    config_file = os.path.expanduser(config_path or "~/.oci/config")
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(config_file):
        return

    if not parser.has_section(profile_name):
        oci_config.pop("security_token_file", None)
        return

    explicit_security_token = False
    current_section: typing.Optional[str] = None
    with open(config_file, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                continue
            if current_section == profile_name and line.split("=", 1)[0].strip() == "security_token_file":
                explicit_security_token = True
                break

    if not explicit_security_token:
        oci_config.pop("security_token_file", None)


def _usage_from_oci(usage_data: typing.Optional[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
    usage_data = usage_data or {}
    input_tokens = usage_data.get("inputTokens", 0)
    output_tokens = usage_data.get("completionTokens", usage_data.get("outputTokens", 0))

    return {
        "tokens": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "billed_units": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    }


def get_event_hooks(
    oci_config: typing.Dict[str, typing.Any],
    oci_region: str,
    oci_compartment_id: str,
    is_v2_client: bool = False,
) -> typing.Dict[str, typing.List[EventHook]]:
    """
    Create httpx event hooks for OCI request/response transformation.

    Args:
        oci_config: OCI configuration dictionary
        oci_region: OCI region (e.g., "us-chicago-1")
        oci_compartment_id: OCI compartment OCID
        is_v2_client: Whether this is for OciClientV2 (True) or OciClient (False)

    Returns:
        Dictionary of event hooks for httpx
    """
    return {
        "request": [
            map_request_to_oci(
                oci_config=oci_config,
                oci_region=oci_region,
                oci_compartment_id=oci_compartment_id,
                is_v2_client=is_v2_client,
            ),
        ],
        "response": [map_response_from_oci()],
    }


def map_request_to_oci(
    oci_config: typing.Dict[str, typing.Any],
    oci_region: str,
    oci_compartment_id: str,
    is_v2_client: bool = False,
) -> EventHook:
    """
    Create event hook that transforms Cohere requests to OCI format and signs them.

    Args:
        oci_config: OCI configuration dictionary
        oci_region: OCI region
        oci_compartment_id: OCI compartment OCID
        is_v2_client: Whether this is for OciClientV2 (True) or OciClient (False)

    Returns:
        Event hook function for httpx
    """
    oci = lazy_oci()

    # Create OCI signer based on config type
    # Priority order: instance/resource principal > session-based auth > API key auth
    if "signer" in oci_config:
        signer = oci_config["signer"]  # Instance/resource principal
    elif "security_token_file" in oci_config:
        # Session-based authentication with security token.
        # The token file is re-read on every request so that OCI CLI token refreshes
        # (e.g. `oci session refresh`) are picked up without restarting the client.
        key_file = oci_config.get("key_file")
        if not key_file:
            raise ValueError(
                "OCI config profile is missing 'key_file'. "
                "Session-based auth requires a key_file entry in your OCI config profile."
            )
        token_file_path = os.path.expanduser(oci_config["security_token_file"])
        private_key = oci.signer.load_private_key_from_file(os.path.expanduser(key_file))

        class _RefreshingSecurityTokenSigner:
            """Wraps SecurityTokenSigner and re-reads the token file before each signing call."""

            def __init__(self) -> None:
                self._token_file = token_file_path
                self._private_key = private_key
                self._refresh()

            def _refresh(self) -> None:
                with open(self._token_file, "r") as _f:
                    _token = _f.read().strip()
                self._signer = oci.auth.signers.SecurityTokenSigner(
                    token=_token,
                    private_key=self._private_key,
                )

            # Delegate all attribute access to the inner signer, refreshing first.
            def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
                self._refresh()
                return self._signer(*args, **kwargs)

            def __getattr__(self, name: str) -> typing.Any:
                if name.startswith("_"):
                    raise AttributeError(name)
                self._refresh()
                return getattr(self._signer, name)

        signer = _RefreshingSecurityTokenSigner()
    elif "user" in oci_config:
        signer = oci.signer.Signer(
            tenancy=oci_config["tenancy"],
            user=oci_config["user"],
            fingerprint=oci_config["fingerprint"],
            private_key_file_location=oci_config.get("key_file"),
            private_key_content=oci_config.get("key_content"),
        )
    else:
        # Config doesn't have user or security token - unsupported
        raise ValueError(
            "OCI config is missing 'user' field and no security_token_file found. "
            "Please use a profile with standard API key authentication, "
            "session-based authentication, or provide direct credentials via oci_user_id parameter."
        )

    def _event_hook(request: httpx.Request) -> None:
        # Extract Cohere API details
        path_parts = request.url.path.split("/")
        endpoint = path_parts[-1]
        body = json.loads(request.read())

        # Build OCI URL
        url = get_oci_url(
            region=oci_region,
            endpoint=endpoint,
        )

        # Transform request body to OCI format
        oci_body = transform_request_to_oci(
            endpoint=endpoint,
            cohere_body=body,
            compartment_id=oci_compartment_id,
            is_v2=is_v2_client,
        )

        # Prepare request for signing
        oci_body_bytes = json.dumps(oci_body).encode("utf-8")

        # Build headers for signing
        headers = {
            "content-type": "application/json",
            "date": email.utils.formatdate(usegmt=True),
        }

        # Create a requests.PreparedRequest for OCI signing
        oci_request = requests.Request(
            method=request.method,
            url=url,
            headers=headers,
            data=oci_body_bytes,
        )
        prepped_request = oci_request.prepare()

        # Sign the request using OCI signer (modifies headers in place)
        signer.do_request_sign(prepped_request)

        # Update httpx request with signed headers
        request.url = URL(url)
        request.headers = httpx.Headers(prepped_request.headers)
        request.stream = ByteStream(oci_body_bytes)
        request._content = oci_body_bytes
        request.extensions["endpoint"] = endpoint
        request.extensions["is_stream"] = body.get("stream", False)
        request.extensions["is_v2"] = is_v2_client

    return _event_hook


def map_response_from_oci() -> EventHook:
    """
    Create event hook that transforms OCI responses to Cohere format.

    Returns:
        Event hook function for httpx
    """

    def _hook(response: httpx.Response) -> None:
        endpoint = response.request.extensions["endpoint"]
        is_stream = response.request.extensions.get("is_stream", False)
        is_v2 = response.request.extensions.get("is_v2", False)

        output: typing.Iterator[bytes]

        # Only transform successful responses (200-299)
        # Let error responses pass through unchanged so SDK error handling works
        if not (200 <= response.status_code < 300):
            return

        # For streaming responses, wrap the stream with a transformer
        if is_stream:
            original_stream = typing.cast(typing.Iterator[bytes], response.stream)
            transformed_stream = transform_oci_stream_wrapper(original_stream, endpoint, is_v2)
            response.stream = Streamer(transformed_stream)
            # Reset consumption flags
            if hasattr(response, "_content"):
                del response._content
            response.is_stream_consumed = False
            response.is_closed = False
            return

        # Handle non-streaming responses
        oci_response = json.loads(response.read())
        cohere_response = transform_oci_response_to_cohere(endpoint, oci_response, is_v2)
        output = iter([json.dumps(cohere_response).encode("utf-8")])

        response.stream = Streamer(output)

        # Reset response for re-reading
        if hasattr(response, "_content"):
            del response._content
        response.is_stream_consumed = False
        response.is_closed = False

    return _hook


def get_oci_url(
    region: str,
    endpoint: str,
) -> str:
    """
    Map Cohere endpoints to OCI Generative AI endpoints.

    Args:
        region: OCI region (e.g., "us-chicago-1")
        endpoint: Cohere endpoint name
    Returns:
        Full OCI Generative AI endpoint URL
    """
    base = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    api_version = "20231130"

    # Map Cohere endpoints to OCI actions
    action_map = {
        "embed": "embedText",
        "chat": "chat",
    }

    action = action_map.get(endpoint)
    if action is None:
        raise ValueError(
            f"Endpoint '{endpoint}' is not supported by OCI Generative AI. "
            f"Supported endpoints: {list(action_map.keys())}"
        )
    return f"{base}/{api_version}/actions/{action}"


def normalize_model_for_oci(model: str) -> str:
    """
    Normalize model name for OCI.

    OCI accepts model names in the format "cohere.model-name" or full OCIDs.
    This function ensures proper formatting for all regions.

    Args:
        model: Model name (e.g., "command-r-08-2024") or full OCID

    Returns:
        Normalized model identifier (e.g., "cohere.command-r-08-2024" or OCID)

    Examples:
        >>> normalize_model_for_oci("command-a-03-2025")
        "cohere.command-a-03-2025"
        >>> normalize_model_for_oci("cohere.embed-english-v3.0")
        "cohere.embed-english-v3.0"
        >>> normalize_model_for_oci("ocid1.generativeaimodel.oc1...")
        "ocid1.generativeaimodel.oc1..."
    """
    if not model:
        raise ValueError("OCI requests require a non-empty model name")

    # If it's already an OCID, return as-is (works across all regions)
    if model.startswith("ocid1."):
        return model

    # Add "cohere." prefix if not present
    if not model.startswith("cohere."):
        return f"cohere.{model}"

    return model


def transform_request_to_oci(
    endpoint: str,
    cohere_body: typing.Dict[str, typing.Any],
    compartment_id: str,
    is_v2: bool = False,
) -> typing.Dict[str, typing.Any]:
    """
    Transform Cohere request body to OCI format.

    Args:
        endpoint: Cohere endpoint name
        cohere_body: Original Cohere request body
        compartment_id: OCI compartment OCID
        is_v2: Whether this request comes from OciClientV2 (True) or OciClient (False)

    Returns:
        Transformed request body in OCI format
    """
    model = normalize_model_for_oci(cohere_body.get("model", ""))

    if endpoint == "embed":
        if "texts" in cohere_body:
            inputs = cohere_body["texts"]
        elif "inputs" in cohere_body:
            inputs = cohere_body["inputs"]
        elif "images" in cohere_body:
            raise ValueError("OCI embed does not support the top-level 'images' parameter; use 'inputs' instead")
        else:
            raise ValueError("OCI embed requires either 'texts' or 'inputs'")

        oci_body = {
            "inputs": inputs,
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model,
            },
            "compartmentId": compartment_id,
        }

        # Add optional fields only if provided
        if "input_type" in cohere_body:
            oci_body["inputType"] = cohere_body["input_type"].upper()

        if "truncate" in cohere_body:
            oci_body["truncate"] = cohere_body["truncate"].upper()

        if "embedding_types" in cohere_body:
            oci_body["embeddingTypes"] = [et.upper() for et in cohere_body["embedding_types"]]
        if "max_tokens" in cohere_body:
            oci_body["maxTokens"] = cohere_body["max_tokens"]
        if "output_dimension" in cohere_body:
            oci_body["outputDimension"] = cohere_body["output_dimension"]
        if "priority" in cohere_body:
            oci_body["priority"] = cohere_body["priority"]

        return oci_body

    elif endpoint == "chat":
        # Validate that the request body matches the client type
        has_messages = "messages" in cohere_body
        has_message = "message" in cohere_body
        if is_v2 and not has_messages:
            raise ValueError(
                "OciClientV2 requires the V2 API format ('messages' array). "
                "Got a V1-style request with 'message' string. "
                "Use OciClient for V1 models like Command R, "
                "or switch to the V2 messages format."
            )
        if not is_v2 and has_messages and not has_message:
            raise ValueError(
                "OciClient uses the V1 API format (single 'message' string). "
                "Got a V2-style request with 'messages' array. "
                "Use OciClientV2 for V2 models like Command A."
            )

        chat_request: typing.Dict[str, typing.Any] = {
            "apiFormat": "COHEREV2" if is_v2 else "COHERE",
        }

        if is_v2:
            # V2: Transform Cohere V2 messages to OCI V2 format
            # Cohere sends: [{"role": "user", "content": "text"}]
            # OCI expects: [{"role": "USER", "content": [{"type": "TEXT", "text": "..."}]}]
            oci_messages = []
            for msg in cohere_body["messages"]:
                oci_msg: typing.Dict[str, typing.Any] = {
                    "role": msg["role"].upper(),
                }

                # Transform content
                if isinstance(msg.get("content"), str):
                    oci_msg["content"] = [{"type": "TEXT", "text": msg["content"]}]
                elif isinstance(msg.get("content"), list):
                    transformed_content = []
                    for item in msg["content"]:
                        if isinstance(item, dict) and "type" in item:
                            transformed_item = item.copy()
                            transformed_item["type"] = item["type"].upper()
                            transformed_content.append(transformed_item)
                        else:
                            transformed_content.append(item)
                    oci_msg["content"] = transformed_content
                else:
                    oci_msg["content"] = msg.get("content") or []

                if "tool_calls" in msg:
                    oci_msg["toolCalls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    oci_msg["toolCallId"] = msg["tool_call_id"]
                if "tool_plan" in msg:
                    oci_msg["toolPlan"] = msg["tool_plan"]

                oci_messages.append(oci_msg)

            chat_request["messages"] = oci_messages

            # V2 optional parameters
            if "max_tokens" in cohere_body:
                chat_request["maxTokens"] = cohere_body["max_tokens"]
            if "temperature" in cohere_body:
                chat_request["temperature"] = cohere_body["temperature"]
            if "k" in cohere_body:
                chat_request["topK"] = cohere_body["k"]
            if "p" in cohere_body:
                chat_request["topP"] = cohere_body["p"]
            if "seed" in cohere_body:
                chat_request["seed"] = cohere_body["seed"]
            if "frequency_penalty" in cohere_body:
                chat_request["frequencyPenalty"] = cohere_body["frequency_penalty"]
            if "presence_penalty" in cohere_body:
                chat_request["presencePenalty"] = cohere_body["presence_penalty"]
            if "stop_sequences" in cohere_body:
                chat_request["stopSequences"] = cohere_body["stop_sequences"]
            if "tools" in cohere_body:
                chat_request["tools"] = cohere_body["tools"]
            if "strict_tools" in cohere_body:
                chat_request["strictTools"] = cohere_body["strict_tools"]
            if "documents" in cohere_body:
                chat_request["documents"] = cohere_body["documents"]
            if "citation_options" in cohere_body:
                chat_request["citationOptions"] = cohere_body["citation_options"]
            if "response_format" in cohere_body:
                chat_request["responseFormat"] = cohere_body["response_format"]
            if "safety_mode" in cohere_body:
                chat_request["safetyMode"] = cohere_body["safety_mode"]
            if "logprobs" in cohere_body:
                chat_request["logprobs"] = cohere_body["logprobs"]
            if "tool_choice" in cohere_body:
                chat_request["toolChoice"] = cohere_body["tool_choice"]
            if "priority" in cohere_body:
                chat_request["priority"] = cohere_body["priority"]
            # Thinking parameter for Command A Reasoning models
            if "thinking" in cohere_body and cohere_body["thinking"] is not None:
                thinking = cohere_body["thinking"]
                oci_thinking: typing.Dict[str, typing.Any] = {}
                if "type" in thinking:
                    oci_thinking["type"] = thinking["type"].upper()
                if "token_budget" in thinking and thinking["token_budget"] is not None:
                    oci_thinking["tokenBudget"] = thinking["token_budget"]
                if oci_thinking:
                    chat_request["thinking"] = oci_thinking
        else:
            # V1: single message string
            chat_request["message"] = cohere_body["message"]

            if "temperature" in cohere_body:
                chat_request["temperature"] = cohere_body["temperature"]
            if "max_tokens" in cohere_body:
                chat_request["maxTokens"] = cohere_body["max_tokens"]
            if "k" in cohere_body:
                chat_request["topK"] = cohere_body["k"]
            if "p" in cohere_body:
                chat_request["topP"] = cohere_body["p"]
            if "seed" in cohere_body:
                chat_request["seed"] = cohere_body["seed"]
            if "stop_sequences" in cohere_body:
                chat_request["stopSequences"] = cohere_body["stop_sequences"]
            if "frequency_penalty" in cohere_body:
                chat_request["frequencyPenalty"] = cohere_body["frequency_penalty"]
            if "presence_penalty" in cohere_body:
                chat_request["presencePenalty"] = cohere_body["presence_penalty"]
            if "preamble" in cohere_body:
                chat_request["preambleOverride"] = cohere_body["preamble"]
            if "chat_history" in cohere_body:
                chat_request["chatHistory"] = cohere_body["chat_history"]
            if "documents" in cohere_body:
                chat_request["documents"] = cohere_body["documents"]
            if "tools" in cohere_body:
                chat_request["tools"] = cohere_body["tools"]
            if "tool_results" in cohere_body:
                chat_request["toolResults"] = cohere_body["tool_results"]
            if "response_format" in cohere_body:
                chat_request["responseFormat"] = cohere_body["response_format"]
            if "safety_mode" in cohere_body:
                chat_request["safetyMode"] = cohere_body["safety_mode"]
            if "priority" in cohere_body:
                chat_request["priority"] = cohere_body["priority"]

        # Handle streaming for both versions
        if cohere_body.get("stream"):
            chat_request["isStream"] = True

        # Top level OCI request structure
        oci_body = {
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model,
            },
            "compartmentId": compartment_id,
            "chatRequest": chat_request,
        }

        return oci_body

    raise ValueError(
        f"Endpoint '{endpoint}' is not supported by OCI Generative AI on-demand inference. "
        "Supported endpoints: ['embed', 'chat']"
    )


def transform_oci_response_to_cohere(
    endpoint: str, oci_response: typing.Dict[str, typing.Any], is_v2: bool = False,
) -> typing.Dict[str, typing.Any]:
    """
    Transform OCI response to Cohere format.

    Args:
        endpoint: Cohere endpoint name
        oci_response: OCI response body
        is_v2: Whether this is a V2 API response

    Returns:
        Transformed response in Cohere format
    """
    if endpoint == "embed":
        embeddings_data = oci_response.get("embeddings", {})

        if isinstance(embeddings_data, dict):
            normalized_embeddings = {str(key).lower(): value for key, value in embeddings_data.items()}
        else:
            normalized_embeddings = {"float": embeddings_data}

        if is_v2:
            embeddings = normalized_embeddings
        else:
            embeddings = normalized_embeddings.get("float", [])

        meta = {
            "api_version": {"version": "1"},
        }
        usage = _usage_from_oci(oci_response.get("usage"))
        if "tokens" in usage:
            meta["tokens"] = usage["tokens"]
        if "billed_units" in usage:
            meta["billed_units"] = usage["billed_units"]

        response_type = "embeddings_by_type" if is_v2 else "embeddings_floats"

        return {
            "response_type": response_type,
            "id": oci_response.get("id", str(uuid.uuid4())),
            "embeddings": embeddings,
            "texts": [],
            "meta": meta,
        }

    elif endpoint == "chat":
        chat_response = oci_response.get("chatResponse", {})

        if is_v2:
            usage = _usage_from_oci(chat_response.get("usage"))
            message = chat_response.get("message", {})

            if "role" in message:
                message = {**message, "role": message["role"].lower()}

            if "content" in message and isinstance(message["content"], list):
                transformed_content = []
                for item in message["content"]:
                    if isinstance(item, dict):
                        transformed_item = item.copy()
                        if "type" in transformed_item:
                            transformed_item["type"] = transformed_item["type"].lower()
                        transformed_content.append(transformed_item)
                    else:
                        transformed_content.append(item)
                message = {**message, "content": transformed_content}

            if "toolCalls" in message:
                tool_calls = message["toolCalls"]
                message = {k: v for k, v in message.items() if k != "toolCalls"}
                message["tool_calls"] = tool_calls
            if "toolPlan" in message:
                tool_plan = message["toolPlan"]
                message = {k: v for k, v in message.items() if k != "toolPlan"}
                message["tool_plan"] = tool_plan

            return {
                "id": chat_response.get("id", str(uuid.uuid4())),
                "message": message,
                "finish_reason": chat_response.get("finishReason", "COMPLETE"),
                "usage": usage,
            }

        # V1 response
        meta = {
            "api_version": {"version": "1"},
        }
        usage = _usage_from_oci(chat_response.get("usage"))
        if "tokens" in usage:
            meta["tokens"] = usage["tokens"]
        if "billed_units" in usage:
            meta["billed_units"] = usage["billed_units"]

        return {
            "text": chat_response.get("text", ""),
            "generation_id": str(uuid.uuid4()),
            "chat_history": chat_response.get("chatHistory", []),
            "finish_reason": chat_response.get("finishReason", "COMPLETE"),
            "citations": chat_response.get("citations", []),
            "documents": chat_response.get("documents", []),
            "search_queries": chat_response.get("searchQueries", []),
            "meta": meta,
        }

    return oci_response


def transform_oci_stream_wrapper(
    stream: typing.Iterator[bytes], endpoint: str, is_v2: bool = False,
) -> typing.Iterator[bytes]:
    """
    Wrap OCI stream and transform events to Cohere format.

    Args:
        stream: Original OCI stream iterator
        endpoint: Cohere endpoint name
        is_v2: Whether this is a V2 API stream

    Yields:
        Bytes of transformed streaming events
    """
    generation_id = str(uuid.uuid4())
    emitted_start = False
    emitted_content_end = False
    current_content_type: typing.Optional[str] = None
    current_content_index = 0
    final_finish_reason = "COMPLETE"
    final_usage: typing.Optional[typing.Dict[str, typing.Any]] = None
    full_v1_text = ""
    final_v1_finish_reason = "COMPLETE"
    buffer = b""

    def _emit_v2_event(event: typing.Dict[str, typing.Any]) -> bytes:
        return b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"

    def _emit_v1_event(event: typing.Dict[str, typing.Any]) -> bytes:
        return json.dumps(event).encode("utf-8") + b"\n"

    def _current_content_type(oci_event: typing.Dict[str, typing.Any]) -> typing.Optional[str]:
        message = oci_event.get("message")
        if isinstance(message, dict):
            content_list = message.get("content")
            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                oci_type = content_list[0].get("type", "TEXT").upper()
                return "thinking" if oci_type == "THINKING" else "text"
        return None  # finish-only or non-content event — don't trigger a type transition

    def _transform_v2_event(oci_event: typing.Dict[str, typing.Any]) -> typing.Iterator[bytes]:
        nonlocal emitted_start, emitted_content_end, current_content_type, current_content_index
        nonlocal final_finish_reason, final_usage

        event_content_type = _current_content_type(oci_event)
        open_type = event_content_type or "text"

        if not emitted_start:
            yield _emit_v2_event(
                {
                    "type": "message-start",
                    "id": generation_id,
                    "delta": {"message": {"role": "assistant"}},
                }
            )
            yield _emit_v2_event(
                {
                    "type": "content-start",
                    "index": current_content_index,
                    "delta": {"message": {"content": {"type": open_type}}},
                }
            )
            emitted_start = True
            current_content_type = open_type
        elif event_content_type is not None and current_content_type != event_content_type:
            yield _emit_v2_event({"type": "content-end", "index": current_content_index})
            current_content_index += 1
            yield _emit_v2_event(
                {
                    "type": "content-start",
                    "index": current_content_index,
                    "delta": {"message": {"content": {"type": event_content_type}}},
                }
            )
            current_content_type = event_content_type
            emitted_content_end = False

        for cohere_event in typing.cast(
            typing.List[typing.Dict[str, typing.Any]], transform_stream_event(endpoint, oci_event, is_v2=True)
        ):
            if "index" in cohere_event:
                cohere_event = {**cohere_event, "index": current_content_index}
            if cohere_event["type"] == "content-end":
                emitted_content_end = True
                final_finish_reason = oci_event.get("finishReason", final_finish_reason)
                final_usage = _usage_from_oci(oci_event.get("usage"))
            yield _emit_v2_event(cohere_event)

    def _transform_v1_event(oci_event: typing.Dict[str, typing.Any]) -> typing.Iterator[bytes]:
        nonlocal emitted_start, full_v1_text, final_v1_finish_reason
        if not emitted_start:
            yield _emit_v1_event({
                "event_type": "stream-start",
                "generation_id": generation_id,
                "is_finished": False,
            })
            emitted_start = True
        event = transform_stream_event(endpoint, oci_event, is_v2=False)
        if isinstance(event, dict):
            if event.get("event_type") == "text-generation" and event.get("text"):
                full_v1_text += typing.cast(str, event["text"])
            if "finishReason" in oci_event:
                final_v1_finish_reason = oci_event.get("finishReason", final_v1_finish_reason)
            yield _emit_v1_event(event)

    def _process_line(line: str) -> typing.Iterator[bytes]:
        if not line.startswith("data: "):
            return

        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            if is_v2:
                if emitted_start:
                    if not emitted_content_end:
                        yield _emit_v2_event({"type": "content-end", "index": current_content_index})
                    message_end_event: typing.Dict[str, typing.Any] = {
                        "type": "message-end",
                        "id": generation_id,
                        "delta": {"finish_reason": final_finish_reason},
                    }
                    if final_usage:
                        message_end_event["delta"]["usage"] = final_usage
                    yield _emit_v2_event(message_end_event)
            else:
                yield _emit_v1_event(
                    {
                        "event_type": "stream-end",
                        "finish_reason": final_v1_finish_reason,
                        "response": {
                            "text": full_v1_text,
                            "generation_id": generation_id,
                            "finish_reason": final_v1_finish_reason,
                        },
                    }
                )
            return

        try:
            oci_event = json.loads(data_str)
        except json.JSONDecodeError:
            return

        try:
            if is_v2:
                for event_bytes in _transform_v2_event(oci_event):
                    yield event_bytes
            else:
                for event_bytes in _transform_v1_event(oci_event):
                    yield event_bytes
        except Exception as exc:
            raise RuntimeError(f"OCI stream event transformation failed for endpoint '{endpoint}': {exc}") from exc

    for chunk in stream:
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8").strip()
            for event_bytes in _process_line(line):
                yield event_bytes

    if buffer.strip():
        line = buffer.decode("utf-8").strip()
        for event_bytes in _process_line(line):
            yield event_bytes


def transform_stream_event(
    endpoint: str, oci_event: typing.Dict[str, typing.Any], is_v2: bool = False,
) -> typing.Union[typing.Dict[str, typing.Any], typing.List[typing.Dict[str, typing.Any]]]:
    """
    Transform individual OCI stream event to Cohere format.

    Args:
        endpoint: Cohere endpoint name
        oci_event: OCI stream event
        is_v2: Whether this is a V2 API stream

    Returns:
        V2: List of transformed events. V1: Single transformed event dict.
    """
    if endpoint == "chat":
        if is_v2:
            content_type = "text"
            content_value = ""
            message = oci_event.get("message")

            if "message" in oci_event and not isinstance(message, dict):
                raise TypeError("OCI V2 stream event message must be an object")

            if isinstance(message, dict) and "content" in message:
                content_list = message["content"]
                if content_list and isinstance(content_list, list) and len(content_list) > 0:
                    first_content = content_list[0]
                    oci_type = first_content.get("type", "TEXT").upper()
                    if oci_type == "THINKING":
                        content_type = "thinking"
                        content_value = first_content.get("thinking", "")
                    else:
                        content_type = "text"
                        content_value = first_content.get("text", "")

            events: typing.List[typing.Dict[str, typing.Any]] = []
            if content_value:
                delta_content: typing.Dict[str, typing.Any] = {}
                if content_type == "thinking":
                    delta_content["thinking"] = content_value
                else:
                    delta_content["text"] = content_value

                events.append(
                    {
                        "type": "content-delta",
                        "index": 0,
                        "delta": {
                            "message": {
                                "content": delta_content,
                            }
                        },
                    }
                )

            if "finishReason" in oci_event:
                events.append(
                    {
                        "type": "content-end",
                        "index": 0,
                    }
                )

            return events

        # V1 stream event
        return {
            "event_type": "text-generation",
            "text": oci_event.get("text", ""),
            "is_finished": oci_event.get("isFinished", False),
        }

    return [] if is_v2 else {}
