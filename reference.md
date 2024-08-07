# Reference
<details><summary><code>client.<a href="src/cohere/base_client.py">chat_stream</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generates a text response to a user message.
To learn how to use the Chat API with Streaming and RAG follow our [Text Generation guides](https://docs.cohere.com/docs/chat-api).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere import (
    ChatConnector,
    ChatStreamRequestConnectorsSearchOptions,
    Message_Chatbot,
    ResponseFormat_Text,
    Tool,
    ToolCall,
    ToolParameterDefinitionsValue,
    ToolResult,
)
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
response = client.chat_stream(
    message="string",
    model="string",
    preamble="string",
    chat_history=[
        Message_Chatbot(
            message="string",
            tool_calls=[
                ToolCall(
                    name="string",
                    parameters={"string": {"key": "value"}},
                )
            ],
        )
    ],
    conversation_id="string",
    prompt_truncation="OFF",
    connectors=[
        ChatConnector(
            id="string",
            user_access_token="string",
            continue_on_failure=True,
            options={"string": {"key": "value"}},
        )
    ],
    search_queries_only=True,
    documents=[{"string": {"key": "value"}}],
    citation_quality="fast",
    temperature=1.1,
    max_tokens=1,
    max_input_tokens=1,
    k=1,
    p=1.1,
    seed=1,
    stop_sequences=["string"],
    connectors_search_options=ChatStreamRequestConnectorsSearchOptions(
        seed=1,
    ),
    frequency_penalty=1.1,
    presence_penalty=1.1,
    raw_prompting=True,
    return_prompt=True,
    tools=[
        Tool(
            name="string",
            description="string",
            parameter_definitions={
                "string": ToolParameterDefinitionsValue(
                    description="string",
                    type="string",
                    required=True,
                )
            },
        )
    ],
    tool_results=[
        ToolResult(
            call=ToolCall(
                name="string",
                parameters={"string": {"key": "value"}},
            ),
            outputs=[{"string": {"key": "value"}}],
        )
    ],
    force_single_step=True,
    response_format=ResponseFormat_Text(),
)
for chunk in response:
    yield chunk

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**message:** `str` 

Text input for the model to respond to.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

Defaults to `command-r-plus`.

The name of a compatible [Cohere model](https://docs.cohere.com/docs/models) or the ID of a [fine-tuned](https://docs.cohere.com/docs/chat-fine-tuning) model.
Compatible Deployments: Cohere Platform, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**preamble:** `typing.Optional[str]` 

When specified, the default Cohere preamble will be replaced with the provided one. Preambles are a part of the prompt used to adjust the model's overall behavior and conversation style, and use the `SYSTEM` role.

The `SYSTEM` role is also used for the contents of the optional `chat_history=` parameter. When used with the `chat_history=` parameter it adds content throughout a conversation. Conversely, when used with the `preamble=` parameter it adds content at the start of the conversation only.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**chat_history:** `typing.Optional[typing.Sequence[Message]]` 

A list of previous messages between the user and the model, giving the model conversational context for responding to the user's `message`.

Each item represents a single message in the chat history, excluding the current user turn. It has two properties: `role` and `message`. The `role` identifies the sender (`CHATBOT`, `SYSTEM`, or `USER`), while the `message` contains the text content.

The chat_history parameter should not be used for `SYSTEM` messages in most cases. Instead, to add a `SYSTEM` role message at the beginning of a conversation, the `preamble` parameter should be used.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**conversation_id:** `typing.Optional[str]` 

An alternative to `chat_history`.

Providing a `conversation_id` creates or resumes a persisted conversation with the specified ID. The ID can be any non empty string.
Compatible Deployments: Cohere Platform

    
</dd>
</dl>

<dl>
<dd>

**prompt_truncation:** `typing.Optional[ChatStreamRequestPromptTruncation]` 

Defaults to `AUTO` when `connectors` are specified and `OFF` in all other cases.

Dictates how the prompt will be constructed.

With `prompt_truncation` set to "AUTO", some elements from `chat_history` and `documents` will be dropped in an attempt to construct a prompt that fits within the model's context length limit. During this process the order of the documents and chat history will be changed and ranked by relevance.

With `prompt_truncation` set to "AUTO_PRESERVE_ORDER", some elements from `chat_history` and `documents` will be dropped in an attempt to construct a prompt that fits within the model's context length limit. During this process the order of the documents and chat history will be preserved as they are inputted into the API.

With `prompt_truncation` set to "OFF", no elements will be dropped. If the sum of the inputs exceeds the model's context length limit, a `TooManyTokens` error will be returned.
Compatible Deployments: Cohere Platform Only AUTO_PRESERVE_ORDER: Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**connectors:** `typing.Optional[typing.Sequence[ChatConnector]]` 

Accepts `{"id": "web-search"}`, and/or the `"id"` for a custom [connector](https://docs.cohere.com/docs/connectors), if you've [created](https://docs.cohere.com/docs/creating-and-deploying-a-connector) one.

When specified, the model's reply will be enriched with information found by querying each of the connectors (RAG).
Compatible Deployments: Cohere Platform

    
</dd>
</dl>

<dl>
<dd>

**search_queries_only:** `typing.Optional[bool]` 

Defaults to `false`.

When `true`, the response will only contain a list of generated search queries, but no search will take place, and no reply from the model to the user's `message` will be generated.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**documents:** `typing.Optional[typing.Sequence[ChatDocument]]` 

A list of relevant documents that the model can cite to generate a more accurate reply. Each document is a string-string dictionary.

Example:
`[
  { "title": "Tall penguins", "text": "Emperor penguins are the tallest." },
  { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica." },
]`

Keys and values from each document will be serialized to a string and passed to the model. The resulting generation will include citations that reference some of these documents.

Some suggested keys are "text", "author", and "date". For better generation quality, it is recommended to keep the total word count of the strings in the dictionary to under 300 words.

An `id` field (string) can be optionally supplied to identify the document in the citations. This field will not be passed to the model.

An `_excludes` field (array of strings) can be optionally supplied to omit some key-value pairs from being shown to the model. The omitted fields will still show up in the citation object. The "_excludes" field will not be passed to the model.

See ['Document Mode'](https://docs.cohere.com/docs/retrieval-augmented-generation-rag#document-mode) in the guide for more information.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**citation_quality:** `typing.Optional[ChatStreamRequestCitationQuality]` 

Defaults to `"accurate"`.

Dictates the approach taken to generating citations as part of the RAG flow by allowing the user to specify whether they want `"accurate"` results, `"fast"` results or no results.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` 

Defaults to `0.3`.

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations, and higher temperatures mean more random generations.

Randomness can be further maximized by increasing the  value of the `p` parameter.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` 

The maximum number of tokens the model will generate as part of the response. Note: Setting a low value may result in incomplete generations.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**max_input_tokens:** `typing.Optional[int]` 

The maximum number of input tokens to send to the model. If not specified, `max_input_tokens` is the model's context length limit minus a small buffer.

Input will be truncated according to the `prompt_truncation` parameter.
Compatible Deployments: Cohere Platform

    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 

Ensures only the top `k` most likely tokens are considered for generation at each step.
Defaults to `0`, min value of `0`, max value of `500`.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**p:** `typing.Optional[float]` 

Ensures that only the most likely tokens, with total probability mass of `p`, are considered for generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
Defaults to `0.75`. min value of `0.01`, max value of `0.99`.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.Sequence[str]]` 

A list of up to 5 strings that the model will use to stop generation. If the model generates a string that matches any of the strings in the list, it will stop generating tokens and return the generated text up to that point not including the stop sequence.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.

Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.

Used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**raw_prompting:** `typing.Optional[bool]` 

When enabled, the user's prompt will be sent to the model without
any pre-processing.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**return_prompt:** `typing.Optional[bool]` — The prompt is returned in the `prompt` response field when this is enabled.
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.Sequence[Tool]]` 

A list of available tools (functions) that the model may suggest invoking before producing a text response.

When `tools` is passed (without `tool_results`), the `text` field in the response will be `""` and the `tool_calls` field in the response will be populated with a list of tool calls that need to be made. If no calls need to be made, the `tool_calls` array will be empty.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**tool_results:** `typing.Optional[typing.Sequence[ToolResult]]` 

A list of results from invoking tools recommended by the model in the previous chat turn. Results are used to produce a text response and will be referenced in citations. When using `tool_results`, `tools` must be passed as well.
Each tool_result contains information about how it was invoked, as well as a list of outputs in the form of dictionaries.

**Note**: `outputs` must be a list of objects. If your tool returns a single object (eg `{"status": 200}`), make sure to wrap it in a list.
```
tool_results = [
  {
    "call": {
      "name": <tool name>,
      "parameters": {
        <param name>: <param value>
      }
    },
    "outputs": [{
      <key>: <value>
    }]
  },
  ...
]
```
**Note**: Chat calls with `tool_results` should not be included in the Chat history to avoid duplication of the message text.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**force_single_step:** `typing.Optional[bool]` — Forces the chat to be single step. Defaults to `false`.
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[ResponseFormat]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">chat</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generates a text response to a user message.
To learn how to use the Chat API with Streaming and RAG follow our [Text Generation guides](https://docs.cohere.com/docs/chat-api).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.chat(
    message="Can you give me a global market overview of solar panels?",
    prompt_truncation="OFF",
    temperature=0.3,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**message:** `str` 

Text input for the model to respond to.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

Defaults to `command-r-plus`.

The name of a compatible [Cohere model](https://docs.cohere.com/docs/models) or the ID of a [fine-tuned](https://docs.cohere.com/docs/chat-fine-tuning) model.
Compatible Deployments: Cohere Platform, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**preamble:** `typing.Optional[str]` 

When specified, the default Cohere preamble will be replaced with the provided one. Preambles are a part of the prompt used to adjust the model's overall behavior and conversation style, and use the `SYSTEM` role.

The `SYSTEM` role is also used for the contents of the optional `chat_history=` parameter. When used with the `chat_history=` parameter it adds content throughout a conversation. Conversely, when used with the `preamble=` parameter it adds content at the start of the conversation only.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**chat_history:** `typing.Optional[typing.Sequence[Message]]` 

A list of previous messages between the user and the model, giving the model conversational context for responding to the user's `message`.

Each item represents a single message in the chat history, excluding the current user turn. It has two properties: `role` and `message`. The `role` identifies the sender (`CHATBOT`, `SYSTEM`, or `USER`), while the `message` contains the text content.

The chat_history parameter should not be used for `SYSTEM` messages in most cases. Instead, to add a `SYSTEM` role message at the beginning of a conversation, the `preamble` parameter should be used.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**conversation_id:** `typing.Optional[str]` 

An alternative to `chat_history`.

Providing a `conversation_id` creates or resumes a persisted conversation with the specified ID. The ID can be any non empty string.
Compatible Deployments: Cohere Platform

    
</dd>
</dl>

<dl>
<dd>

**prompt_truncation:** `typing.Optional[ChatRequestPromptTruncation]` 

Defaults to `AUTO` when `connectors` are specified and `OFF` in all other cases.

Dictates how the prompt will be constructed.

With `prompt_truncation` set to "AUTO", some elements from `chat_history` and `documents` will be dropped in an attempt to construct a prompt that fits within the model's context length limit. During this process the order of the documents and chat history will be changed and ranked by relevance.

With `prompt_truncation` set to "AUTO_PRESERVE_ORDER", some elements from `chat_history` and `documents` will be dropped in an attempt to construct a prompt that fits within the model's context length limit. During this process the order of the documents and chat history will be preserved as they are inputted into the API.

With `prompt_truncation` set to "OFF", no elements will be dropped. If the sum of the inputs exceeds the model's context length limit, a `TooManyTokens` error will be returned.
Compatible Deployments: Cohere Platform Only AUTO_PRESERVE_ORDER: Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**connectors:** `typing.Optional[typing.Sequence[ChatConnector]]` 

Accepts `{"id": "web-search"}`, and/or the `"id"` for a custom [connector](https://docs.cohere.com/docs/connectors), if you've [created](https://docs.cohere.com/docs/creating-and-deploying-a-connector) one.

When specified, the model's reply will be enriched with information found by querying each of the connectors (RAG).
Compatible Deployments: Cohere Platform

    
</dd>
</dl>

<dl>
<dd>

**search_queries_only:** `typing.Optional[bool]` 

Defaults to `false`.

When `true`, the response will only contain a list of generated search queries, but no search will take place, and no reply from the model to the user's `message` will be generated.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**documents:** `typing.Optional[typing.Sequence[ChatDocument]]` 

A list of relevant documents that the model can cite to generate a more accurate reply. Each document is a string-string dictionary.

Example:
`[
  { "title": "Tall penguins", "text": "Emperor penguins are the tallest." },
  { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica." },
]`

Keys and values from each document will be serialized to a string and passed to the model. The resulting generation will include citations that reference some of these documents.

Some suggested keys are "text", "author", and "date". For better generation quality, it is recommended to keep the total word count of the strings in the dictionary to under 300 words.

An `id` field (string) can be optionally supplied to identify the document in the citations. This field will not be passed to the model.

An `_excludes` field (array of strings) can be optionally supplied to omit some key-value pairs from being shown to the model. The omitted fields will still show up in the citation object. The "_excludes" field will not be passed to the model.

See ['Document Mode'](https://docs.cohere.com/docs/retrieval-augmented-generation-rag#document-mode) in the guide for more information.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**citation_quality:** `typing.Optional[ChatRequestCitationQuality]` 

Defaults to `"accurate"`.

Dictates the approach taken to generating citations as part of the RAG flow by allowing the user to specify whether they want `"accurate"` results, `"fast"` results or no results.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` 

Defaults to `0.3`.

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations, and higher temperatures mean more random generations.

Randomness can be further maximized by increasing the  value of the `p` parameter.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` 

The maximum number of tokens the model will generate as part of the response. Note: Setting a low value may result in incomplete generations.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**max_input_tokens:** `typing.Optional[int]` 

The maximum number of input tokens to send to the model. If not specified, `max_input_tokens` is the model's context length limit minus a small buffer.

Input will be truncated according to the `prompt_truncation` parameter.
Compatible Deployments: Cohere Platform

    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 

Ensures only the top `k` most likely tokens are considered for generation at each step.
Defaults to `0`, min value of `0`, max value of `500`.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**p:** `typing.Optional[float]` 

Ensures that only the most likely tokens, with total probability mass of `p`, are considered for generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
Defaults to `0.75`. min value of `0.01`, max value of `0.99`.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.Sequence[str]]` 

A list of up to 5 strings that the model will use to stop generation. If the model generates a string that matches any of the strings in the list, it will stop generating tokens and return the generated text up to that point not including the stop sequence.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.

Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.

Used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**raw_prompting:** `typing.Optional[bool]` 

When enabled, the user's prompt will be sent to the model without
any pre-processing.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**return_prompt:** `typing.Optional[bool]` — The prompt is returned in the `prompt` response field when this is enabled.
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.Sequence[Tool]]` 

A list of available tools (functions) that the model may suggest invoking before producing a text response.

When `tools` is passed (without `tool_results`), the `text` field in the response will be `""` and the `tool_calls` field in the response will be populated with a list of tool calls that need to be made. If no calls need to be made, the `tool_calls` array will be empty.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**tool_results:** `typing.Optional[typing.Sequence[ToolResult]]` 

A list of results from invoking tools recommended by the model in the previous chat turn. Results are used to produce a text response and will be referenced in citations. When using `tool_results`, `tools` must be passed as well.
Each tool_result contains information about how it was invoked, as well as a list of outputs in the form of dictionaries.

**Note**: `outputs` must be a list of objects. If your tool returns a single object (eg `{"status": 200}`), make sure to wrap it in a list.
```
tool_results = [
  {
    "call": {
      "name": <tool name>,
      "parameters": {
        <param name>: <param value>
      }
    },
    "outputs": [{
      <key>: <value>
    }]
  },
  ...
]
```
**Note**: Chat calls with `tool_results` should not be included in the Chat history to avoid duplication of the message text.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**force_single_step:** `typing.Optional[bool]` — Forces the chat to be single step. Defaults to `false`.
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[ResponseFormat]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">generate_stream</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

> 🚧 Warning
>
> This API is marked as "Legacy" and is no longer maintained. Follow the [migration guide](/docs/migrating-from-cogenerate-to-cochat) to start using the Chat API.

Generates realistic text conditioned on a given input.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
response = client.generate_stream(
    prompt="string",
    model="string",
    num_generations=1,
    max_tokens=1,
    truncate="NONE",
    temperature=1.1,
    seed=1,
    preset="string",
    end_sequences=["string"],
    stop_sequences=["string"],
    k=1,
    p=1.1,
    frequency_penalty=1.1,
    presence_penalty=1.1,
    return_likelihoods="GENERATION",
    raw_prompting=True,
)
for chunk in response:
    yield chunk

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**prompt:** `str` 

The input text that serves as the starting point for generating the response.
Note: The prompt will be pre-processed and modified before reaching the model.

    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

The identifier of the model to generate with. Currently available models are `command` (default), `command-nightly` (experimental), `command-light`, and `command-light-nightly` (experimental).
Smaller, "light" models are faster, while larger models will perform better. [Custom models](/docs/training-custom-models) can also be supplied with their full ID.
    
</dd>
</dl>

<dl>
<dd>

**num_generations:** `typing.Optional[int]` — The maximum number of generations that will be returned. Defaults to `1`, min value of `1`, max value of `5`.

    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` 

The maximum number of tokens the model will generate as part of the response. Note: Setting a low value may result in incomplete generations.

This parameter is off by default, and if it's not specified, the model will continue generating until it emits an EOS completion token. See [BPE Tokens](/bpe-tokens-wiki) for more details.

Can only be set to `0` if `return_likelihoods` is set to `ALL` to get the likelihood of the prompt.

    
</dd>
</dl>

<dl>
<dd>

**truncate:** `typing.Optional[GenerateStreamRequestTruncate]` 

One of `NONE|START|END` to specify how the API will handle inputs longer than the maximum token length.

Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.

If `NONE` is selected, when the input exceeds the maximum input token length an error will be returned.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` 

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations. See [Temperature](/temperature-wiki) for more details.
Defaults to `0.75`, min value of `0.0`, max value of `5.0`.

    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**preset:** `typing.Optional[str]` 

Identifier of a custom preset. A preset is a combination of parameters, such as prompt, temperature etc. You can create presets in the [playground](https://dashboard.cohere.com/playground/generate).
When a preset is specified, the `prompt` parameter becomes optional, and any included parameters will override the preset's parameters.

    
</dd>
</dl>

<dl>
<dd>

**end_sequences:** `typing.Optional[typing.Sequence[str]]` — The generated text will be cut at the beginning of the earliest occurrence of an end sequence. The sequence will be excluded from the text.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.Sequence[str]]` — The generated text will be cut at the end of the earliest occurrence of a stop sequence. The sequence will be included the text.
    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 

Ensures only the top `k` most likely tokens are considered for generation at each step.
Defaults to `0`, min value of `0`, max value of `500`.

    
</dd>
</dl>

<dl>
<dd>

**p:** `typing.Optional[float]` 

Ensures that only the most likely tokens, with total probability mass of `p`, are considered for generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
Defaults to `0.75`. min value of `0.01`, max value of `0.99`.

    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` 

Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.

Using `frequency_penalty` in combination with `presence_penalty` is not supported on newer models.

    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.

Can be used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.

Using `frequency_penalty` in combination with `presence_penalty` is not supported on newer models.

    
</dd>
</dl>

<dl>
<dd>

**return_likelihoods:** `typing.Optional[GenerateStreamRequestReturnLikelihoods]` 

One of `GENERATION|ALL|NONE` to specify how and if the token likelihoods are returned with the response. Defaults to `NONE`.

If `GENERATION` is selected, the token likelihoods will only be provided for generated text.

If `ALL` is selected, the token likelihoods will be provided both for the prompt and the generated text.
    
</dd>
</dl>

<dl>
<dd>

**raw_prompting:** `typing.Optional[bool]` — When enabled, the user's prompt will be sent to the model without any pre-processing.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">generate</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

> 🚧 Warning
>
> This API is marked as "Legacy" and is no longer maintained. Follow the [migration guide](/docs/migrating-from-cogenerate-to-cochat) to start using the Chat API.

Generates realistic text conditioned on a given input.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.generate(
    prompt="Please explain to me how LLMs work",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**prompt:** `str` 

The input text that serves as the starting point for generating the response.
Note: The prompt will be pre-processed and modified before reaching the model.

    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

The identifier of the model to generate with. Currently available models are `command` (default), `command-nightly` (experimental), `command-light`, and `command-light-nightly` (experimental).
Smaller, "light" models are faster, while larger models will perform better. [Custom models](/docs/training-custom-models) can also be supplied with their full ID.
    
</dd>
</dl>

<dl>
<dd>

**num_generations:** `typing.Optional[int]` — The maximum number of generations that will be returned. Defaults to `1`, min value of `1`, max value of `5`.

    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` 

The maximum number of tokens the model will generate as part of the response. Note: Setting a low value may result in incomplete generations.

This parameter is off by default, and if it's not specified, the model will continue generating until it emits an EOS completion token. See [BPE Tokens](/bpe-tokens-wiki) for more details.

Can only be set to `0` if `return_likelihoods` is set to `ALL` to get the likelihood of the prompt.

    
</dd>
</dl>

<dl>
<dd>

**truncate:** `typing.Optional[GenerateRequestTruncate]` 

One of `NONE|START|END` to specify how the API will handle inputs longer than the maximum token length.

Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.

If `NONE` is selected, when the input exceeds the maximum input token length an error will be returned.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` 

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations. See [Temperature](/temperature-wiki) for more details.
Defaults to `0.75`, min value of `0.0`, max value of `5.0`.

    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.
Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments

    
</dd>
</dl>

<dl>
<dd>

**preset:** `typing.Optional[str]` 

Identifier of a custom preset. A preset is a combination of parameters, such as prompt, temperature etc. You can create presets in the [playground](https://dashboard.cohere.com/playground/generate).
When a preset is specified, the `prompt` parameter becomes optional, and any included parameters will override the preset's parameters.

    
</dd>
</dl>

<dl>
<dd>

**end_sequences:** `typing.Optional[typing.Sequence[str]]` — The generated text will be cut at the beginning of the earliest occurrence of an end sequence. The sequence will be excluded from the text.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.Sequence[str]]` — The generated text will be cut at the end of the earliest occurrence of a stop sequence. The sequence will be included the text.
    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 

Ensures only the top `k` most likely tokens are considered for generation at each step.
Defaults to `0`, min value of `0`, max value of `500`.

    
</dd>
</dl>

<dl>
<dd>

**p:** `typing.Optional[float]` 

Ensures that only the most likely tokens, with total probability mass of `p`, are considered for generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
Defaults to `0.75`. min value of `0.01`, max value of `0.99`.

    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` 

Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.

Using `frequency_penalty` in combination with `presence_penalty` is not supported on newer models.

    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.

Can be used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.

Using `frequency_penalty` in combination with `presence_penalty` is not supported on newer models.

    
</dd>
</dl>

<dl>
<dd>

**return_likelihoods:** `typing.Optional[GenerateRequestReturnLikelihoods]` 

One of `GENERATION|ALL|NONE` to specify how and if the token likelihoods are returned with the response. Defaults to `NONE`.

If `GENERATION` is selected, the token likelihoods will only be provided for generated text.

If `ALL` is selected, the token likelihoods will be provided both for the prompt and the generated text.
    
</dd>
</dl>

<dl>
<dd>

**raw_prompting:** `typing.Optional[bool]` — When enabled, the user's prompt will be sent to the model without any pre-processing.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">embed</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint returns text embeddings. An embedding is a list of floating point numbers that captures semantic information about the text that it represents.

Embeddings can be used to create text classifiers as well as empower semantic search. To learn more about embeddings, see the embedding page.

If you want to learn more how to use the embedding model, have a look at the [Semantic Search Guide](/docs/semantic-search).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.embed(
    texts=["string"],
    model="string",
    input_type="search_document",
    embedding_types=["float"],
    truncate="NONE",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**texts:** `typing.Sequence[str]` — An array of strings for the model to embed. Maximum number of texts per call is `96`. We recommend reducing the length of each text to be under `512` tokens for optimal quality.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

Defaults to embed-english-v2.0

The identifier of the model. Smaller "light" models are faster, while larger models will perform better. [Custom models](/docs/training-custom-models) can also be supplied with their full ID.

Available models and corresponding embedding dimensions:

* `embed-english-v3.0`  1024
* `embed-multilingual-v3.0`  1024
* `embed-english-light-v3.0`  384
* `embed-multilingual-light-v3.0`  384

* `embed-english-v2.0`  4096
* `embed-english-light-v2.0`  1024
* `embed-multilingual-v2.0`  768
    
</dd>
</dl>

<dl>
<dd>

**input_type:** `typing.Optional[EmbedInputType]` 
    
</dd>
</dl>

<dl>
<dd>

**embedding_types:** `typing.Optional[typing.Sequence[EmbeddingType]]` 

Specifies the types of embeddings you want to get back. Not required and default is None, which returns the Embed Floats response type. Can be one or more of the following types.

* `"float"`: Use this when you want to get back the default float embeddings. Valid for all models.
* `"int8"`: Use this when you want to get back signed int8 embeddings. Valid for only v3 models.
* `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Valid for only v3 models.
* `"binary"`: Use this when you want to get back signed binary embeddings. Valid for only v3 models.
* `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Valid for only v3 models.
    
</dd>
</dl>

<dl>
<dd>

**truncate:** `typing.Optional[EmbedRequestTruncate]` 

One of `NONE|START|END` to specify how the API will handle inputs longer than the maximum token length.

Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.

If `NONE` is selected, when the input exceeds the maximum input token length an error will be returned.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">rerank</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint takes in a query and a list of texts and produces an ordered array with each text assigned a relevance score.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.rerank(
    model="rerank-english-v3.0",
    query="What is the capital of the United States?",
    documents=[
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**query:** `str` — The search query
    
</dd>
</dl>

<dl>
<dd>

**documents:** `typing.Sequence[RerankRequestDocumentsItem]` 

A list of document objects or strings to rerank.
If a document is provided the text fields is required and all other fields will be preserved in the response.

The total max chunks (length of documents * max_chunks_per_doc) must be less than 10000.

We recommend a maximum of 1,000 documents for optimal endpoint performance.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` — The identifier of the model to use, one of : `rerank-english-v3.0`, `rerank-multilingual-v3.0`, `rerank-english-v2.0`, `rerank-multilingual-v2.0`
    
</dd>
</dl>

<dl>
<dd>

**top_n:** `typing.Optional[int]` — The number of most relevant documents or indices to return, defaults to the length of the documents
    
</dd>
</dl>

<dl>
<dd>

**rank_fields:** `typing.Optional[typing.Sequence[str]]` — If a JSON object is provided, you can specify which keys you would like to have considered for reranking. The model will rerank based on order of the fields passed in (i.e. rank_fields=['title','author','text'] will rerank using the values in title, author, text  sequentially. If the length of title, author, and text exceeds the context length of the model, the chunking will not re-consider earlier fields). If not provided, the model will use the default text field for ranking.
    
</dd>
</dl>

<dl>
<dd>

**return_documents:** `typing.Optional[bool]` 

- If false, returns results without the doc text - the api will return a list of {index, relevance score} where index is inferred from the list passed into the request.
- If true, returns results with the doc text passed in - the api will return an ordered list of {index, text, relevance score} where index + text refers to the list passed into the request.
    
</dd>
</dl>

<dl>
<dd>

**max_chunks_per_doc:** `typing.Optional[int]` — The maximum number of chunks to produce internally from a document
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">classify</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint makes a prediction about which label fits the specified text inputs best. To make a prediction, Classify uses the provided `examples` of text + label pairs as a reference.
Note: [Fine-tuned models](https://docs.cohere.com/docs/classify-fine-tuning) trained on classification examples don't require the `examples` parameter to be passed in explicitly.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere import ClassifyExample
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.classify(
    inputs=["Confirm your email address", "hey i need u to send some $"],
    examples=[
        ClassifyExample(
            text="Dermatologists don't like her!",
            label="Spam",
        ),
        ClassifyExample(
            text="Hello, open to this?",
            label="Spam",
        ),
        ClassifyExample(
            text="I need help please wire me $1000 right now",
            label="Spam",
        ),
        ClassifyExample(
            text="Nice to know you ;)",
            label="Spam",
        ),
        ClassifyExample(
            text="Please help me?",
            label="Spam",
        ),
        ClassifyExample(
            text="Your parcel will be delivered today",
            label="Not spam",
        ),
        ClassifyExample(
            text="Review changes to our Terms and Conditions",
            label="Not spam",
        ),
        ClassifyExample(
            text="Weekly sync notes",
            label="Not spam",
        ),
        ClassifyExample(
            text="Re: Follow up from today’s meeting",
            label="Not spam",
        ),
        ClassifyExample(
            text="Pre-read for tomorrow",
            label="Not spam",
        ),
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**inputs:** `typing.Sequence[str]` 

A list of up to 96 texts to be classified. Each one must be a non-empty string.
There is, however, no consistent, universal limit to the length a particular input can be. We perform classification on the first `x` tokens of each input, and `x` varies depending on which underlying model is powering classification. The maximum token length for each model is listed in the "max tokens" column [here](https://docs.cohere.com/docs/models).
Note: by default the `truncate` parameter is set to `END`, so tokens exceeding the limit will be automatically dropped. This behavior can be disabled by setting `truncate` to `NONE`, which will result in validation errors for longer texts.
    
</dd>
</dl>

<dl>
<dd>

**examples:** `typing.Optional[typing.Sequence[ClassifyExample]]` 

An array of examples to provide context to the model. Each example is a text string and its associated label/class. Each unique label requires at least 2 examples associated with it; the maximum number of examples is 2500, and each example has a maximum length of 512 tokens. The values should be structured as `{text: "...",label: "..."}`.
Note: [Fine-tuned Models](https://docs.cohere.com/docs/classify-fine-tuning) trained on classification examples don't require the `examples` parameter to be passed in explicitly.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` — The identifier of the model. Currently available models are `embed-multilingual-v2.0`, `embed-english-light-v2.0`, and `embed-english-v2.0` (default). Smaller "light" models are faster, while larger models will perform better. [Fine-tuned models](https://docs.cohere.com/docs/fine-tuning) can also be supplied with their full ID.
    
</dd>
</dl>

<dl>
<dd>

**preset:** `typing.Optional[str]` — The ID of a custom playground preset. You can create presets in the [playground](https://dashboard.cohere.com/playground/classify?model=large). If you use a preset, all other parameters become optional, and any included parameters will override the preset's parameters.
    
</dd>
</dl>

<dl>
<dd>

**truncate:** `typing.Optional[ClassifyRequestTruncate]` 

One of `NONE|START|END` to specify how the API will handle inputs longer than the maximum token length.
Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
If `NONE` is selected, when the input exceeds the maximum input token length an error will be returned.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">summarize</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

> 🚧 Warning
>
> This API is marked as "Legacy" and is no longer maintained. Follow the [migration guide](/docs/migrating-from-cogenerate-to-cochat) to start using the Chat API.

Generates a summary in English for a given text.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.summarize(
    text='Ice cream is a sweetened frozen food typically eaten as a snack or dessert. It may be made from milk or cream and is flavoured with a sweetener, either sugar or an alternative, and a spice, such as cocoa or vanilla, or with fruit such as strawberries or peaches. It can also be made by whisking a flavored cream base and liquid nitrogen together. Food coloring is sometimes added, in addition to stabilizers. The mixture is cooled below the freezing point of water and stirred to incorporate air spaces and to prevent detectable ice crystals from forming. The result is a smooth, semi-solid foam that is solid at very low temperatures (below 2 °C or 35 °F). It becomes more malleable as its temperature increases.\n\nThe meaning of the name "ice cream" varies from one country to another. In some countries, such as the United States, "ice cream" applies only to a specific variety, and most governments regulate the commercial use of the various terms according to the relative quantities of the main ingredients, notably the amount of cream. Products that do not meet the criteria to be called ice cream are sometimes labelled "frozen dairy dessert" instead. In other countries, such as Italy and Argentina, one word is used fo\r all variants. Analogues made from dairy alternatives, such as goat\'s or sheep\'s milk, or milk substitutes (e.g., soy, cashew, coconut, almond milk or tofu), are available for those who are lactose intolerant, allergic to dairy protein or vegan.',
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**text:** `str` — The text to generate a summary for. Can be up to 100,000 characters long. Currently the only supported language is English.
    
</dd>
</dl>

<dl>
<dd>

**length:** `typing.Optional[SummarizeRequestLength]` — One of `short`, `medium`, `long`, or `auto` defaults to `auto`. Indicates the approximate length of the summary. If `auto` is selected, the best option will be picked based on the input text.
    
</dd>
</dl>

<dl>
<dd>

**format:** `typing.Optional[SummarizeRequestFormat]` — One of `paragraph`, `bullets`, or `auto`, defaults to `auto`. Indicates the style in which the summary will be delivered - in a free form paragraph or in bullet points. If `auto` is selected, the best option will be picked based on the input text.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` — The identifier of the model to generate the summary with. Currently available models are `command` (default), `command-nightly` (experimental), `command-light`, and `command-light-nightly` (experimental). Smaller, "light" models are faster, while larger models will perform better.
    
</dd>
</dl>

<dl>
<dd>

**extractiveness:** `typing.Optional[SummarizeRequestExtractiveness]` — One of `low`, `medium`, `high`, or `auto`, defaults to `auto`. Controls how close to the original text the summary is. `high` extractiveness summaries will lean towards reusing sentences verbatim, while `low` extractiveness summaries will tend to paraphrase more. If `auto` is selected, the best option will be picked based on the input text.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` — Ranges from 0 to 5. Controls the randomness of the output. Lower values tend to generate more “predictable” output, while higher values tend to generate more “creative” output. The sweet spot is typically between 0 and 1.
    
</dd>
</dl>

<dl>
<dd>

**additional_command:** `typing.Optional[str]` — A free-form instruction for modifying how the summaries get generated. Should complete the sentence "Generate a summary _". Eg. "focusing on the next steps" or "written by Yoda"
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">tokenize</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint splits input text into smaller units called tokens using byte-pair encoding (BPE). To learn more about tokenization and byte pair encoding, see the tokens page.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.tokenize(
    text="tokenize me! :D",
    model="command",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**text:** `str` — The string to be tokenized, the minimum text length is 1 character, and the maximum text length is 65536 characters.
    
</dd>
</dl>

<dl>
<dd>

**model:** `str` — An optional parameter to provide the model name. This will ensure that the tokenization uses the tokenizer used by that model.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">detokenize</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint takes tokens using byte-pair encoding and returns their text representation. To learn more about tokenization and byte pair encoding, see the tokens page.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.detokenize(
    tokens=[10104, 12221, 1315, 34, 1420, 69],
    model="command",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**tokens:** `typing.Sequence[int]` — The list of tokens to be detokenized.
    
</dd>
</dl>

<dl>
<dd>

**model:** `str` — An optional parameter to provide the model name. This will ensure that the detokenization is done by the tokenizer used by that model.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.<a href="src/cohere/base_client.py">check_api_key</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Checks that the api key in the Authorization header is valid and active
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.check_api_key()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## V2
<details><summary><code>client.v2.<a href="src/cohere/v2/client.py">chat_stream</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere import (
    ChatMessage2_Assistant,
    Citation,
    Source_Tool,
    TextContent,
    Tool2,
    Tool2Function,
    ToolCall2,
    ToolCall2Function,
    V2ChatStreamRequestResponseFormat,
)
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
response = client.v2.chat_stream(
    model="string",
    messages=[
        ChatMessage2_Assistant(
            tool_calls=[
                ToolCall2(
                    id="string",
                    function=ToolCall2Function(
                        name="string",
                        arguments="string",
                    ),
                )
            ],
            tool_plan="string",
            content=[
                TextContent(
                    text="string",
                )
            ],
            citations=[
                Citation(
                    start="string",
                    end="string",
                    text="string",
                    sources=[
                        Source_Tool(
                            id="string",
                            tool_output={"string": {"key": "value"}},
                        )
                    ],
                )
            ],
        )
    ],
    tools=[
        Tool2(
            function=Tool2Function(
                name="string",
                description="string",
                parameters={"string": {"key": "value"}},
            ),
        )
    ],
    tool_choice="AUTO",
    citation_mode="FAST",
    truncation_mode="OFF",
    response_format=V2ChatStreamRequestResponseFormat(
        schema={"string": {"key": "value"}},
    ),
    max_tokens=1,
    stop_sequences=["string"],
    max_input_tokens=1,
    temperature=1.1,
    seed=1,
    frequency_penalty=1.1,
    presence_penalty=1.1,
    k=1,
    p=1,
    return_prompt=True,
)
for chunk in response:
    yield chunk

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**model:** `str` — The model to use for the chat.
    
</dd>
</dl>

<dl>
<dd>

**messages:** `ChatMessages` 
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.Sequence[Tool2]]` 
    
</dd>
</dl>

<dl>
<dd>

**tool_choice:** `typing.Optional[V2ChatStreamRequestToolChoice]` 
    
</dd>
</dl>

<dl>
<dd>

**citation_mode:** `typing.Optional[V2ChatStreamRequestCitationMode]` 
    
</dd>
</dl>

<dl>
<dd>

**truncation_mode:** `typing.Optional[V2ChatStreamRequestTruncationMode]` 
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[V2ChatStreamRequestResponseFormat]` 
    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` — The maximum number of tokens to generate.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.Sequence[str]]` — A list of strings that the model will stop generating at.
    
</dd>
</dl>

<dl>
<dd>

**max_input_tokens:** `typing.Optional[int]` — The maximum number of tokens to feed into the model.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` — The temperature of the model.
    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` — The frequency penalty of the model.
    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` — The presence penalty of the model.
    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**p:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**return_prompt:** `typing.Optional[bool]` — Whether to return the prompt in the response.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.v2.<a href="src/cohere/v2/client.py">chat</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.v2.chat(
    model="model",
    messages=[],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**model:** `str` — The model to use for the chat.
    
</dd>
</dl>

<dl>
<dd>

**messages:** `ChatMessages` 
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.Sequence[Tool2]]` 
    
</dd>
</dl>

<dl>
<dd>

**tool_choice:** `typing.Optional[V2ChatRequestToolChoice]` 
    
</dd>
</dl>

<dl>
<dd>

**citation_mode:** `typing.Optional[V2ChatRequestCitationMode]` 
    
</dd>
</dl>

<dl>
<dd>

**truncation_mode:** `typing.Optional[V2ChatRequestTruncationMode]` 
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[V2ChatRequestResponseFormat]` 
    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` — The maximum number of tokens to generate.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.Sequence[str]]` — A list of strings that the model will stop generating at.
    
</dd>
</dl>

<dl>
<dd>

**max_input_tokens:** `typing.Optional[int]` — The maximum number of tokens to feed into the model.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` — The temperature of the model.
    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` — The frequency penalty of the model.
    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` — The presence penalty of the model.
    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**p:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**return_prompt:** `typing.Optional[bool]` — Whether to return the prompt in the response.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## EmbedJobs
<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">list</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

The list embed job endpoint allows users to view all embed jobs history for that specific user.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.embed_jobs.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This API launches an async Embed job for a [Dataset](https://docs.cohere.com/docs/datasets) of type `embed-input`. The result of a completed embed job is new Dataset of type `embed-output`, which contains the original text entries and the corresponding embeddings.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.embed_jobs.create(
    model="model",
    dataset_id="dataset_id",
    input_type="search_document",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**model:** `str` 

ID of the embedding model.

Available models and corresponding embedding dimensions:

- `embed-english-v3.0` : 1024
- `embed-multilingual-v3.0` : 1024
- `embed-english-light-v3.0` : 384
- `embed-multilingual-light-v3.0` : 384

    
</dd>
</dl>

<dl>
<dd>

**dataset_id:** `str` — ID of a [Dataset](https://docs.cohere.com/docs/datasets). The Dataset must be of type `embed-input` and must have a validation status `Validated`
    
</dd>
</dl>

<dl>
<dd>

**input_type:** `EmbedInputType` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — The name of the embed job.
    
</dd>
</dl>

<dl>
<dd>

**embedding_types:** `typing.Optional[typing.Sequence[EmbeddingType]]` 

Specifies the types of embeddings you want to get back. Not required and default is None, which returns the Embed Floats response type. Can be one or more of the following types.

* `"float"`: Use this when you want to get back the default float embeddings. Valid for all models.
* `"int8"`: Use this when you want to get back signed int8 embeddings. Valid for only v3 models.
* `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Valid for only v3 models.
* `"binary"`: Use this when you want to get back signed binary embeddings. Valid for only v3 models.
* `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Valid for only v3 models.
    
</dd>
</dl>

<dl>
<dd>

**truncate:** `typing.Optional[CreateEmbedJobRequestTruncate]` 

One of `START|END` to specify how the API will handle inputs longer than the maximum token length.

Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.

    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This API retrieves the details about an embed job started by the same user.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.embed_jobs.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The ID of the embed job to retrieve.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">cancel</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This API allows users to cancel an active embed job. Once invoked, the embedding process will be terminated, and users will be charged for the embeddings processed up to the cancellation point. It's important to note that partial results will not be available to users after cancellation.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.embed_jobs.cancel(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The ID of the embed job to cancel.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Datasets
<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List datasets that have been created.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.datasets.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_type:** `typing.Optional[str]` — optional filter by dataset type
    
</dd>
</dl>

<dl>
<dd>

**before:** `typing.Optional[dt.datetime]` — optional filter before a date
    
</dd>
</dl>

<dl>
<dd>

**after:** `typing.Optional[dt.datetime]` — optional filter after a date
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[float]` — optional limit to number of results
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[float]` — optional offset to start of results
    
</dd>
</dl>

<dl>
<dd>

**validation_status:** `typing.Optional[DatasetValidationStatus]` — optional filter by validation status
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset by uploading a file. See ['Dataset Creation'](https://docs.cohere.com/docs/datasets#dataset-creation) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.datasets.create(
    name="name",
    type="embed-input",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — The name of the uploaded dataset.
    
</dd>
</dl>

<dl>
<dd>

**type:** `DatasetType` — The dataset type, which is used to validate the data. Valid types are `embed-input`, `reranker-finetune-input`, `single-label-classification-finetune-input`, `chat-finetune-input`, and `multi-label-classification-finetune-input`.
    
</dd>
</dl>

<dl>
<dd>

**data:** `from __future__ import annotations

core.File` — See core.File for more documentation
    
</dd>
</dl>

<dl>
<dd>

**keep_original_file:** `typing.Optional[bool]` — Indicates if the original file should be stored.
    
</dd>
</dl>

<dl>
<dd>

**skip_malformed_input:** `typing.Optional[bool]` — Indicates whether rows with malformed input should be dropped (instead of failing the validation check). Dropped rows will be returned in the warnings field.
    
</dd>
</dl>

<dl>
<dd>

**keep_fields:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — List of names of fields that will be persisted in the Dataset. By default the Dataset will retain only the required fields indicated in the [schema for the corresponding Dataset type](https://docs.cohere.com/docs/datasets#dataset-types). For example, datasets of type `embed-input` will drop all fields other than the required `text` field. If any of the fields in `keep_fields` are missing from the uploaded file, Dataset validation will fail.
    
</dd>
</dl>

<dl>
<dd>

**optional_fields:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — List of names of fields that will be persisted in the Dataset. By default the Dataset will retain only the required fields indicated in the [schema for the corresponding Dataset type](https://docs.cohere.com/docs/datasets#dataset-types). For example, Datasets of type `embed-input` will drop all fields other than the required `text` field. If any of the fields in `optional_fields` are missing from the uploaded file, Dataset validation will pass.
    
</dd>
</dl>

<dl>
<dd>

**text_separator:** `typing.Optional[str]` — Raw .txt uploads will be split into entries using the text_separator value.
    
</dd>
</dl>

<dl>
<dd>

**csv_delimiter:** `typing.Optional[str]` — The delimiter used for .csv uploads.
    
</dd>
</dl>

<dl>
<dd>

**dry_run:** `typing.Optional[bool]` — flag to enable dry_run mode
    
</dd>
</dl>

<dl>
<dd>

**eval_data:** `from __future__ import annotations

typing.Optional[core.File]` — See core.File for more documentation
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">get_usage</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

View the dataset storage usage for your Organization. Each Organization can have up to 10GB of storage across all their users.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.datasets.get_usage()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve a dataset by ID. See ['Datasets'](https://docs.cohere.com/docs/datasets) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.datasets.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a dataset by ID. Datasets are automatically deleted after 30 days, but they can also be deleted manually.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.datasets.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Connectors
<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a list of connectors ordered by descending creation date (newer first). See ['Managing your Connector'](https://docs.cohere.com/docs/managing-your-connector) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.connectors.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[float]` — Maximum number of connectors to return [0, 100].
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[float]` — Number of connectors to skip before returning results [0, inf].
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new connector. The connector is tested during registration and will cancel registration when the test is unsuccessful. See ['Creating and Deploying a Connector'](https://docs.cohere.com/docs/creating-and-deploying-a-connector) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.connectors.create(
    name="name",
    url="url",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — A human-readable name for the connector.
    
</dd>
</dl>

<dl>
<dd>

**url:** `str` — The URL of the connector that will be used to search for documents.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — A description of the connector.
    
</dd>
</dl>

<dl>
<dd>

**excludes:** `typing.Optional[typing.Sequence[str]]` — A list of fields to exclude from the prompt (fields remain in the document).
    
</dd>
</dl>

<dl>
<dd>

**oauth:** `typing.Optional[CreateConnectorOAuth]` — The OAuth 2.0 configuration for the connector. Cannot be specified if service_auth is specified.
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` — Whether the connector is active or not.
    
</dd>
</dl>

<dl>
<dd>

**continue_on_failure:** `typing.Optional[bool]` — Whether a chat request should continue or not if the request to this connector fails.
    
</dd>
</dl>

<dl>
<dd>

**service_auth:** `typing.Optional[CreateConnectorServiceAuth]` — The service to service authentication configuration for the connector. Cannot be specified if oauth is specified.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve a connector by ID. See ['Connectors'](https://docs.cohere.com/docs/connectors) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.connectors.get(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The ID of the connector to retrieve.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a connector by ID. See ['Connectors'](https://docs.cohere.com/docs/connectors) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.connectors.delete(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The ID of the connector to delete.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">update</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a connector by ID. Omitted fields will not be updated. See ['Managing your Connector'](https://docs.cohere.com/docs/managing-your-connector) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.connectors.update(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The ID of the connector to update.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — A human-readable name for the connector.
    
</dd>
</dl>

<dl>
<dd>

**url:** `typing.Optional[str]` — The URL of the connector that will be used to search for documents.
    
</dd>
</dl>

<dl>
<dd>

**excludes:** `typing.Optional[typing.Sequence[str]]` — A list of fields to exclude from the prompt (fields remain in the document).
    
</dd>
</dl>

<dl>
<dd>

**oauth:** `typing.Optional[CreateConnectorOAuth]` — The OAuth 2.0 configuration for the connector. Cannot be specified if service_auth is specified.
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**continue_on_failure:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**service_auth:** `typing.Optional[CreateConnectorServiceAuth]` — The service to service authentication configuration for the connector. Cannot be specified if oauth is specified.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">o_auth_authorize</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Authorize the connector with the given ID for the connector oauth app. See ['Connector Authentication'](https://docs.cohere.com/docs/connector-authentication) for more information.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.connectors.o_auth_authorize(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The ID of the connector to authorize.
    
</dd>
</dl>

<dl>
<dd>

**after_token_redirect:** `typing.Optional[str]` — The URL to redirect to after the connector has been authorized.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Models
<details><summary><code>client.models.<a href="src/cohere/models/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the details of a model, provided its name.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.models.get(
    model="command-r",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**model:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/cohere/models/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a list of models available for use. The list contains models from Cohere as well as your fine-tuned models.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.models.list()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page_size:** `typing.Optional[float]` 

Maximum number of models to include in a page
Defaults to `20`, min value of `1`, max value of `1000`.
    
</dd>
</dl>

<dl>
<dd>

**page_token:** `typing.Optional[str]` — Page token provided in the `next_page_token` field of a previous response.
    
</dd>
</dl>

<dl>
<dd>

**endpoint:** `typing.Optional[CompatibleEndpoint]` — When provided, filters the list of models to only those that are compatible with the specified endpoint.
    
</dd>
</dl>

<dl>
<dd>

**default_only:** `typing.Optional[bool]` — When provided, filters the list of models to only the default model to the endpoint. This parameter is only valid when `endpoint` is provided.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## /finetuning
<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">list_finetuned_models</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.list_finetuned_models()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page_size:** `typing.Optional[int]` — Maximum number of results to be returned by the server. If 0, defaults to 50.
    
</dd>
</dl>

<dl>
<dd>

**page_token:** `typing.Optional[str]` — Request a specific page of the list results.
    
</dd>
</dl>

<dl>
<dd>

**order_by:** `typing.Optional[str]` 

Comma separated list of fields. For example: "created_at,name". The default
sorting order is ascending. To specify descending order for a field, append
" desc" to the field name. For example: "created_at desc,name".

Supported sorting fields:

- created_at (default)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">create_finetuned_model</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client
from cohere.finetuning import BaseModel, FinetunedModel, Settings

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.create_finetuned_model(
    request=FinetunedModel(
        name="api-test",
        settings=Settings(
            base_model=BaseModel(
                base_type="BASE_TYPE_GENERATIVE",
            ),
            dataset_id="my-dataset-id",
        ),
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `FinetunedModel` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">get_finetuned_model</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.get_finetuned_model(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The fine-tuned model ID.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">delete_finetuned_model</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.delete_finetuned_model(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — The fine-tuned model ID.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">update_finetuned_model</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client
from cohere.finetuning import BaseModel, Settings

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.update_finetuned_model(
    id="id",
    name="name",
    settings=Settings(
        base_model=BaseModel(
            base_type="BASE_TYPE_UNSPECIFIED",
        ),
        dataset_id="dataset_id",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — FinetunedModel ID.
    
</dd>
</dl>

<dl>
<dd>

**name:** `str` — FinetunedModel name (e.g. `foobar`).
    
</dd>
</dl>

<dl>
<dd>

**settings:** `Settings` — FinetunedModel settings such as dataset, hyperparameters...
    
</dd>
</dl>

<dl>
<dd>

**creator_id:** `typing.Optional[str]` — User ID of the creator.
    
</dd>
</dl>

<dl>
<dd>

**organization_id:** `typing.Optional[str]` — Organization ID.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[Status]` — Current stage in the life-cycle of the fine-tuned model.
    
</dd>
</dl>

<dl>
<dd>

**created_at:** `typing.Optional[dt.datetime]` — Creation timestamp.
    
</dd>
</dl>

<dl>
<dd>

**updated_at:** `typing.Optional[dt.datetime]` — Latest update timestamp.
    
</dd>
</dl>

<dl>
<dd>

**completed_at:** `typing.Optional[dt.datetime]` — Timestamp for the completed fine-tuning.
    
</dd>
</dl>

<dl>
<dd>

**last_used:** `typing.Optional[dt.datetime]` — Timestamp for the latest request to this fine-tuned model.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">list_events</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.list_events(
    finetuned_model_id="finetuned_model_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**finetuned_model_id:** `str` — The parent fine-tuned model ID.
    
</dd>
</dl>

<dl>
<dd>

**page_size:** `typing.Optional[int]` — Maximum number of results to be returned by the server. If 0, defaults to 50.
    
</dd>
</dl>

<dl>
<dd>

**page_token:** `typing.Optional[str]` — Request a specific page of the list results.
    
</dd>
</dl>

<dl>
<dd>

**order_by:** `typing.Optional[str]` 

Comma separated list of fields. For example: "created_at,name". The default
sorting order is ascending. To specify descending order for a field, append
" desc" to the field name. For example: "created_at desc,name".

Supported sorting fields:

- created_at (default)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">list_training_step_metrics</a>(...)</code></summary>
<dl>
<dd>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from cohere.client import Client

client = Client(
    client_name="YOUR_CLIENT_NAME",
    token="YOUR_TOKEN",
)
client.finetuning.list_training_step_metrics(
    finetuned_model_id="finetuned_model_id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**finetuned_model_id:** `str` — The parent fine-tuned model ID.
    
</dd>
</dl>

<dl>
<dd>

**page_size:** `typing.Optional[int]` — Maximum number of results to be returned by the server. If 0, defaults to 50.
    
</dd>
</dl>

<dl>
<dd>

**page_token:** `typing.Optional[str]` — Request a specific page of the list results.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

