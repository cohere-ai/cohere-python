# Cohere Python SDK

![](banner.png)

[![version badge](https://img.shields.io/pypi/v/cohere)](https://pypi.org/project/cohere/)
![license badge](https://img.shields.io/github/license/cohere-ai/cohere-python)
[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-SDK%20generated%20by%20Fern-brightgreen)](https://github.com/fern-api/fern)

The Cohere Python SDK allows access to Cohere models across many different platforms: the cohere platform, AWS (Bedrock, Sagemaker), Azure, GCP and Oracle OCI. For a full list of support and snippets, please take a look at the [SDK support docs page](https://docs.cohere.com/docs/cohere-works-everywhere).

## Documentation

Cohere documentation and API reference is available [here](https://docs.cohere.com/).

## Installation

```
pip install cohere
```

## Usage

```Python
import cohere

co = cohere.ClientV2()

response = co.chat(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "hello world!"}],
)

print(response)
```

> [!TIP]
> You can set a system environment variable `CO_API_KEY` to avoid writing your api key within your code, e.g. add `export CO_API_KEY=theapikeyforyouraccount`
> in your ~/.zshrc or ~/.bashrc, open a new terminal, then code calling `cohere.Client()` will read this key.


## Streaming

The SDK supports streaming endpoints. To take advantage of this feature for chat,
use `chat_stream`.

```Python
import cohere

co = cohere.ClientV2()

response = co.chat_stream(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "hello world!"}],
)

for event in response:
    if event.type == "content-delta":
        print(event.delta.message.content.text, end="")
```

## Oracle Cloud Infrastructure (OCI)

The SDK supports Oracle Cloud Infrastructure (OCI) Generative AI service. First, install the OCI SDK:

```
pip install 'cohere[oci]'
```

Then use the `OciClient` or `OciClientV2`:

```Python
import cohere

# Using OCI config file authentication (default: ~/.oci/config)
co = cohere.OciClient(
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)

response = co.embed(
    model="embed-english-v3.0",
    texts=["Hello world"],
    input_type="search_document",
)

print(response.embeddings)
```

### OCI Authentication Methods

**1. Config File (Default)**
```Python
co = cohere.OciClient(
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
    # Uses ~/.oci/config with DEFAULT profile
)
```

**2. Custom Profile**
```Python
co = cohere.OciClient(
    oci_profile="MY_PROFILE",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

**3. Session-based Authentication (Security Token)**
```Python
# Works with OCI CLI session tokens
co = cohere.OciClient(
    oci_profile="MY_SESSION_PROFILE",  # Profile with security_token_file
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

**4. Direct Credentials**
```Python
co = cohere.OciClient(
    oci_user_id="ocid1.user.oc1...",
    oci_fingerprint="xx:xx:xx:...",
    oci_tenancy_id="ocid1.tenancy.oc1...",
    oci_private_key_path="~/.oci/key.pem",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

**5. Instance Principal (for OCI Compute instances)**
```Python
co = cohere.OciClient(
    auth_type="instance_principal",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

### Supported OCI APIs

The OCI client supports the following Cohere APIs:
- **Embed**: Full support for all embedding models (embed-english-v3.0, embed-light-v3.0, embed-multilingual-v3.0)
- **Chat**: Full support with both V1 (`OciClient`) and V2 (`OciClientV2`) APIs
  - Streaming available via `chat_stream()`
  - Supports Command-R and Command-A model families

### OCI Model Availability and Limitations

**Available on OCI On-Demand Inference:**
- ✅ **Embed models**: embed-english-v3.0, embed-light-v3.0, embed-multilingual-v3.0
- ✅ **Chat models**: command-r-08-2024, command-r-plus, command-a-03-2025

**Not Available on OCI On-Demand Inference:**
- ❌ **Generate API**: OCI TEXT_GENERATION models are base models that require fine-tuning before deployment
- ❌ **Rerank API**: OCI TEXT_RERANK models are base models that require fine-tuning before deployment
- ❌ **Multiple Embedding Types**: OCI on-demand models only support single embedding type per request (cannot request both `float` and `int8` simultaneously)

**Note**: To use Generate or Rerank models on OCI, you need to:
1. Fine-tune the base model using OCI's fine-tuning service
2. Deploy the fine-tuned model to a dedicated endpoint
3. Update your code to use the deployed model endpoint

For the latest model availability, see the [OCI Generative AI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm).

## Contributing

While we value open-source contributions to this SDK, the code is generated programmatically. Additions made directly would have to be moved over to our generation code, otherwise they would be overwritten upon the next generated release. Feel free to open a PR as a proof of concept, but know that we will not be able to merge it as-is. We suggest opening an issue first to discuss with us!

On the other hand, contributions to the README are always very welcome!
