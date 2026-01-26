"""Oracle Cloud Infrastructure (OCI) client for Cohere API."""

import email.utils
import json
import typing
import uuid

import httpx
import requests
from . import (
    EmbedResponse,
    GenerateStreamedResponse,
    Generation,
    NonStreamedChatResponse,
    RerankResponse,
    StreamedChatResponse,
)
from .client import Client, ClientEnvironment
from .client_v2 import ClientV2
from .manually_maintained.lazy_oci_deps import lazy_oci
from httpx import URL, ByteStream, SyncByteStream


class OciClient(Client):
    """
    Cohere client for Oracle Cloud Infrastructure (OCI) Generative AI service.

    Supports all authentication methods:
    - Config file (default): Uses ~/.oci/config
    - Direct credentials: Pass OCI credentials directly
    - Instance principal: For OCI compute instances
    - Resource principal: For OCI functions

    Example using config file:
        ```python
        import cohere

        client = cohere.OciClient(
            oci_region="us-chicago-1",
            oci_compartment_id="ocid1.compartment.oc1...",
        )

        response = client.embed(
            model="embed-english-v3.0",
            texts=["Hello world"],
        )
        ```

    Example using direct credentials:
        ```python
        client = cohere.OciClient(
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
        client = cohere.OciClient(
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
        Client.__init__(
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
                ),
                timeout=timeout,
            ),
        )


class OciClientV2(ClientV2):
    """
    Cohere V2 client for Oracle Cloud Infrastructure (OCI) Generative AI service.

    See OciClient for usage examples and authentication methods.
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
                ),
                timeout=timeout,
            ),
        )


EventHook = typing.Callable[..., typing.Any]


# Response type mappings
response_mapping: typing.Dict[str, typing.Any] = {
    "chat": NonStreamedChatResponse,
    "embed": EmbedResponse,
    "generate": Generation,
    "rerank": RerankResponse,
}

stream_response_mapping: typing.Dict[str, typing.Any] = {
    "chat": StreamedChatResponse,
    "generate": GenerateStreamedResponse,
}


class Streamer(SyncByteStream):
    """Wraps an iterator of bytes for streaming responses."""

    lines: typing.Iterator[bytes]

    def __init__(self, lines: typing.Iterator[bytes]):
        self.lines = lines

    def __iter__(self) -> typing.Iterator[bytes]:
        return self.lines


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
        # Direct credentials provided
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
        return oci.config.from_file(
            file_location=config_path or "~/.oci/config", profile_name=profile or "DEFAULT"
        )


def get_event_hooks(
    oci_config: typing.Dict[str, typing.Any],
    oci_region: str,
    oci_compartment_id: str,
) -> typing.Dict[str, typing.List[EventHook]]:
    """
    Create httpx event hooks for OCI request/response transformation.

    Args:
        oci_config: OCI configuration dictionary
        oci_region: OCI region (e.g., "us-chicago-1")
        oci_compartment_id: OCI compartment OCID

    Returns:
        Dictionary of event hooks for httpx
    """
    return {
        "request": [
            map_request_to_oci(
                oci_config=oci_config,
                oci_region=oci_region,
                oci_compartment_id=oci_compartment_id,
            ),
        ],
        "response": [map_response_from_oci()],
    }


def map_request_to_oci(
    oci_config: typing.Dict[str, typing.Any],
    oci_region: str,
    oci_compartment_id: str,
) -> EventHook:
    """
    Create event hook that transforms Cohere requests to OCI format and signs them.

    Args:
        oci_config: OCI configuration dictionary
        oci_region: OCI region
        oci_compartment_id: OCI compartment OCID

    Returns:
        Event hook function for httpx
    """
    oci = lazy_oci()

    # Create OCI signer based on config type
    if "signer" in oci_config:
        signer = oci_config["signer"]  # Instance/resource principal
    elif "user" not in oci_config:
        # Config doesn't have user - might be session-based or security token based
        # Raise error with helpful message
        raise ValueError(
            "OCI config is missing 'user' field. "
            "Please use a profile with standard API key authentication, "
            "or provide direct credentials via oci_user_id parameter. "
            "Current profile may be using session or security token authentication which is not yet supported."
        )
    else:
        # Config has user field - standard API key auth
        signer = oci.signer.Signer(
            tenancy=oci_config["tenancy"],
            user=oci_config["user"],
            fingerprint=oci_config["fingerprint"],
            private_key_file_location=oci_config.get("key_file"),
            private_key_content=oci_config.get("key_content"),
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
            stream="stream" in endpoint or body.get("stream", False),
        )

        # Transform request body to OCI format
        oci_body = transform_request_to_oci(
            endpoint=endpoint,
            cohere_body=body,
            compartment_id=oci_compartment_id,
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
        request.headers.clear()
        request.headers.update(prepped_request.headers)
        request.stream = ByteStream(oci_body_bytes)
        request._content = oci_body_bytes
        request.extensions["endpoint"] = endpoint
        request.extensions["cohere_body"] = body
        request.extensions["is_stream"] = "stream" in endpoint or body.get("stream", False)

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

        output: typing.Iterator[bytes]

        # Only transform successful responses (200-299)
        # Let error responses pass through unchanged so SDK error handling works
        if not (200 <= response.status_code < 300):
            return

        # For streaming responses, wrap the stream with a transformer
        if is_stream:
            original_stream = response.stream
            transformed_stream = transform_oci_stream_wrapper(original_stream, endpoint)
            response.stream = Streamer(transformed_stream)
            # Reset consumption flags
            if hasattr(response, "_content"):
                del response._content
            response.is_stream_consumed = False
            response.is_closed = False
            return

        # Handle non-streaming responses
        oci_response = json.loads(response.read())
        cohere_response = transform_oci_response_to_cohere(endpoint, oci_response)
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
    stream: bool = False,
) -> str:
    """
    Map Cohere endpoints to OCI Generative AI endpoints.

    Args:
        region: OCI region (e.g., "us-chicago-1")
        endpoint: Cohere endpoint name
        stream: Whether this is a streaming request

    Returns:
        Full OCI Generative AI endpoint URL
    """
    base = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    api_version = "20231130"

    # Map Cohere endpoints to OCI actions
    action_map = {
        "embed": "embedText",
        "chat": "chat",
        "chat_stream": "chat",
        "generate": "generateText",
        "generate_stream": "generateText",
        "rerank": "rerankText",  # OCI uses rerankText, not rerank
    }

    action = action_map.get(endpoint, endpoint)
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
) -> typing.Dict[str, typing.Any]:
    """
    Transform Cohere request body to OCI format.

    Args:
        endpoint: Cohere endpoint name
        cohere_body: Original Cohere request body
        compartment_id: OCI compartment OCID

    Returns:
        Transformed request body in OCI format
    """
    model = normalize_model_for_oci(cohere_body.get("model", ""))

    if endpoint == "embed":
        # Transform Cohere input_type to OCI format
        # Cohere uses: "search_document", "search_query", "classification", "clustering"
        # OCI uses: "SEARCH_DOCUMENT", "SEARCH_QUERY", "CLASSIFICATION", "CLUSTERING"

        oci_body = {
            "inputs": cohere_body["texts"],
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

        return oci_body

    elif endpoint in ["chat", "chat_stream"]:
        # Detect V1 vs V2 API based on request body structure
        is_v2 = "messages" in cohere_body  # V2 uses messages array

        # OCI uses a nested chatRequest structure
        chat_request = {
            "apiFormat": "COHEREV2" if is_v2 else "COHERE",
        }

        if is_v2:
            # V2 API: uses messages array
            # Transform Cohere V2 messages to OCI V2 format
            # Cohere sends: [{"role": "user", "content": "text"}]
            # OCI expects: [{"role": "USER", "content": [{"type": "TEXT", "text": "..."}]}]
            oci_messages = []
            for msg in cohere_body["messages"]:
                oci_msg = {
                    "role": msg["role"].upper(),
                }

                # Transform content
                if isinstance(msg.get("content"), str):
                    # Simple string content -> wrap in array
                    oci_msg["content"] = [{"type": "TEXT", "text": msg["content"]}]
                elif isinstance(msg.get("content"), list):
                    # Already array format (from tool calls, etc.)
                    oci_msg["content"] = msg["content"]
                else:
                    oci_msg["content"] = msg.get("content", [])

                # Add tool_calls if present
                if "tool_calls" in msg:
                    oci_msg["toolCalls"] = msg["tool_calls"]

                oci_messages.append(oci_msg)

            chat_request["messages"] = oci_messages

            # V2 optional parameters (use Cohere's camelCase names for OCI)
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
            if "documents" in cohere_body:
                chat_request["documents"] = cohere_body["documents"]
            if "citation_options" in cohere_body:
                chat_request["citationOptions"] = cohere_body["citation_options"]
            if "safety_mode" in cohere_body:
                chat_request["safetyMode"] = cohere_body["safety_mode"]
        else:
            # V1 API: uses single message string
            chat_request["message"] = cohere_body["message"]

            # V1 optional parameters
            if "temperature" in cohere_body:
                chat_request["temperature"] = cohere_body["temperature"]
            if "max_tokens" in cohere_body:
                chat_request["maxTokens"] = cohere_body["max_tokens"]
            if "preamble" in cohere_body:
                chat_request["preambleOverride"] = cohere_body["preamble"]
            if "chat_history" in cohere_body:
                chat_request["chatHistory"] = cohere_body["chat_history"]

        # Handle streaming for both versions
        if "stream" in endpoint or cohere_body.get("stream"):
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

    elif endpoint in ["generate", "generate_stream"]:
        oci_body = {
            "prompt": cohere_body["prompt"],
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model,
            },
            "compartmentId": compartment_id,
            "isStream": endpoint == "generate_stream" or cohere_body.get("stream", False),
        }
        if "max_tokens" in cohere_body:
            oci_body["maxTokens"] = cohere_body["max_tokens"]
        if "temperature" in cohere_body:
            oci_body["temperature"] = cohere_body["temperature"]
        return oci_body

    elif endpoint == "rerank":
        # OCI rerank uses a flat structure (not nested like chat)
        # and "input" instead of "query"
        oci_body = {
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": model,
            },
            "compartmentId": compartment_id,
            "input": cohere_body["query"],  # OCI uses "input" not "query"
            "documents": cohere_body["documents"],
        }

        # Add optional rerank parameters
        if "top_n" in cohere_body:
            oci_body["topN"] = cohere_body["top_n"]
        if "max_chunks_per_doc" in cohere_body:
            oci_body["maxChunksPerDocument"] = cohere_body["max_chunks_per_doc"]

        return oci_body

    return cohere_body


def transform_oci_response_to_cohere(
    endpoint: str, oci_response: typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:
    """
    Transform OCI response to Cohere format.

    Args:
        endpoint: Cohere endpoint name
        oci_response: OCI response body

    Returns:
        Transformed response in Cohere format
    """
    if endpoint == "embed":
        # OCI returns embeddings in "embeddings" field, may have multiple types
        embeddings_data = oci_response.get("embeddings", {})
        # For now, handle float embeddings (most common case)
        embeddings = embeddings_data.get("float", []) if isinstance(embeddings_data, dict) else embeddings_data

        # Build proper meta structure
        meta = {
            "api_version": {"version": "1"},
        }

        # Add usage info if available
        if "usage" in oci_response and oci_response["usage"]:
            usage = oci_response["usage"]
            # OCI usage has inputTokens, outputTokens, totalTokens
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)

            meta["billed_units"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            meta["tokens"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return {
            "id": oci_response.get("id", str(uuid.uuid4())),
            "embeddings": embeddings,
            "texts": [],  # OCI doesn't return texts
            "meta": meta,
        }

    elif endpoint == "chat" or endpoint == "chat_stream":
        chat_response = oci_response.get("chatResponse", {})

        # Detect V2 response (has apiFormat field)
        is_v2 = chat_response.get("apiFormat") == "COHEREV2"

        if is_v2:
            # V2 response transformation
            # Extract usage for V2
            usage_data = chat_response.get("usage", {})
            usage = {
                "tokens": {
                    "input_tokens": usage_data.get("inputTokens", 0),
                    "output_tokens": usage_data.get("completionTokens", 0),
                },
            }
            if usage_data.get("inputTokens") or usage_data.get("completionTokens"):
                usage["billed_units"] = {
                    "input_tokens": usage_data.get("inputTokens", 0),
                    "output_tokens": usage_data.get("completionTokens", 0),
                }

            return {
                "id": chat_response.get("id", str(uuid.uuid4())),
                "message": chat_response.get("message", {}),
                "finish_reason": chat_response.get("finishReason", "COMPLETE").lower(),
                "usage": usage,
            }
        else:
            # V1 response transformation
            # Build proper meta structure
            meta = {
                "api_version": {"version": "1"},
            }

            # Add usage info if available
            if "usage" in chat_response and chat_response["usage"]:
                usage = chat_response["usage"]
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)

                meta["billed_units"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                meta["tokens"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

            return {
                "text": chat_response.get("text", ""),
                "generation_id": oci_response.get("modelId", str(uuid.uuid4())),
                "chat_history": chat_response.get("chatHistory", []),
                "finish_reason": chat_response.get("finishReason", "COMPLETE"),
                "citations": chat_response.get("citations", []),
                "documents": chat_response.get("documents", []),
                "search_queries": chat_response.get("searchQueries", []),
                "meta": meta,
            }

    elif endpoint == "generate":
        return {
            "id": str(uuid.uuid4()),
            "generations": [
                {
                    "id": str(uuid.uuid4()),
                    "text": oci_response.get("inferenceResponse", {}).get("generatedText", ""),
                    "finish_reason": oci_response.get("finishReason"),
                }
            ],
            "prompt": "",
            "meta": {"api_version": {"version": "1"}},
        }

    elif endpoint == "rerank":
        # OCI returns flat structure with document_ranks
        document_ranks = oci_response.get("documentRanks", [])

        return {
            "id": oci_response.get("id", str(uuid.uuid4())),
            "results": [
                {
                    "index": r.get("index"),
                    "relevance_score": r.get("relevanceScore"),
                }
                for r in document_ranks
            ],
            "meta": {"api_version": {"version": "1"}},
        }

    return oci_response


def transform_oci_stream_wrapper(
    stream: typing.Iterator[bytes], endpoint: str
) -> typing.Iterator[bytes]:
    """
    Wrap OCI stream and transform events to Cohere format.

    Args:
        stream: Original OCI stream iterator
        endpoint: Cohere endpoint name

    Yields:
        Bytes of transformed streaming events
    """
    buffer = b""
    for chunk in stream:
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8").strip()

            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str.strip() == "[DONE]":
                    # Return (not break) to stop the generator completely, preventing further chunk processing
                    return

                try:
                    oci_event = json.loads(data_str)
                    cohere_event = transform_stream_event(endpoint, oci_event)
                    yield json.dumps(cohere_event).encode("utf-8") + b"\n"
                except json.JSONDecodeError:
                    continue


def transform_oci_stream_response(
    response: httpx.Response, endpoint: str
) -> typing.Iterator[bytes]:
    """
    Transform OCI streaming responses to Cohere streaming format.

    OCI uses Server-Sent Events (SSE) format.

    Args:
        response: httpx Response object
        endpoint: Cohere endpoint name

    Yields:
        Bytes of transformed streaming events
    """
    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[6:]  # Remove "data: " prefix
            if data_str.strip() == "[DONE]":
                break

            try:
                oci_event = json.loads(data_str)
                cohere_event = transform_stream_event(endpoint, oci_event)
                yield json.dumps(cohere_event).encode("utf-8") + b"\n"
            except json.JSONDecodeError:
                continue


def transform_stream_event(
    endpoint: str, oci_event: typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:
    """
    Transform individual OCI stream event to Cohere format.

    Args:
        endpoint: Cohere endpoint name
        oci_event: OCI stream event

    Returns:
        Transformed event in Cohere format
    """
    if endpoint in ["chat_stream", "chat"]:
        return {
            "event_type": "text-generation",
            "text": oci_event.get("text", ""),
            "is_finished": oci_event.get("isFinished", False),
        }

    elif endpoint in ["generate_stream", "generate"]:
        return {
            "event_type": "text-generation",
            "text": oci_event.get("text", ""),
            "is_finished": oci_event.get("isFinished", False),
        }

    return oci_event
