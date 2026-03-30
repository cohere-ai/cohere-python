# Reference
<details><summary><code>client.<a href="src/cohere/client.py">chat_stream</a>(...) -> typing.Iterator[bytes]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generates a streamed text response to a user message.

To learn how to use the Chat API and RAG follow our [Text Generation guides](https://docs.cohere.com/docs/chat-api).
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.chat_stream(
    model="command-a-03-2025",
    message="hello!",
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

**stream:** `typing.Literal` 

Defaults to `false`.

When `true`, the response will be a JSON stream of events. The final event will contain the complete response, and will have an `event_type` of `"stream-end"`.

Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.

Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments
    
</dd>
</dl>

<dl>
<dd>

**accepts:** `typing.Optional[typing.Literal]` — Pass text/event-stream to receive the streamed response as server-sent events. The default is `\n` delimited events.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

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

**chat_history:** `typing.Optional[typing.List[Message]]` 

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

Compatible Deployments:
 - AUTO: Cohere Platform Only
 - AUTO_PRESERVE_ORDER: Azure, AWS Sagemaker/Bedrock, Private Deployments
    
</dd>
</dl>

<dl>
<dd>

**connectors:** `typing.Optional[typing.List[ChatConnector]]` 

Accepts `{"id": "web-search"}`, and/or the `"id"` for a custom [connector](https://docs.cohere.com/docs/connectors), if you've [created](https://docs.cohere.com/v1/docs/creating-and-deploying-a-connector) one.

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

**documents:** `typing.Optional[typing.List[ChatDocument]]` 

A list of relevant documents that the model can cite to generate a more accurate reply. Each document is a string-string dictionary.

Example:
```
[
  { "title": "Tall penguins", "text": "Emperor penguins are the tallest." },
  { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica." },
]
```

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

Defaults to `"enabled"`.
Citations are enabled by default for models that support it, but can be turned off by setting `"type": "disabled"`.

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

**stop_sequences:** `typing.Optional[typing.List[str]]` 

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

**tools:** `typing.Optional[typing.List[Tool]]` 

A list of available tools (functions) that the model may suggest invoking before producing a text response.

When `tools` is passed (without `tool_results`), the `text` field in the response will be `""` and the `tool_calls` field in the response will be populated with a list of tool calls that need to be made. If no calls need to be made, the `tool_calls` array will be empty.

Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments
    
</dd>
</dl>

<dl>
<dd>

**tool_results:** `typing.Optional[typing.List[ToolResult]]` 

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

**safety_mode:** `typing.Optional[ChatStreamRequestSafetyMode]` 

Used to select the [safety instruction](https://docs.cohere.com/docs/safety-modes) inserted into the prompt. Defaults to `CONTEXTUAL`.
When `NONE` is specified, the safety instruction will be omitted.

Safety modes are not yet configurable in combination with `tools`, `tool_results` and `documents` parameters.

**Note**: This parameter is only compatible newer Cohere models, starting with [Command R 08-2024](https://docs.cohere.com/docs/command-r#august-2024-release) and [Command R+ 08-2024](https://docs.cohere.com/docs/command-r-plus#august-2024-release).

**Note**: `command-r7b-12-2024` and newer models only support `"CONTEXTUAL"` and `"STRICT"` modes.

Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments
    
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

<details><summary><code>client.<a href="src/cohere/client.py">chat</a>(...) -> NonStreamedChatResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generates a text response to a user message.
To learn how to use the Chat API and RAG follow our [Text Generation guides](https://docs.cohere.com/docs/chat-api).
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.chat_stream(
    model="command-a-03-2025",
    message="Tell me about LLMs",
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

**stream:** `typing.Literal` 

Defaults to `false`.

When `true`, the response will be a JSON stream of events. The final event will contain the complete response, and will have an `event_type` of `"stream-end"`.

Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.

Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments
    
</dd>
</dl>

<dl>
<dd>

**accepts:** `typing.Optional[typing.Literal]` — Pass text/event-stream to receive the streamed response as server-sent events. The default is `\n` delimited events.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

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

**chat_history:** `typing.Optional[typing.List[Message]]` 

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

Compatible Deployments:
 - AUTO: Cohere Platform Only
 - AUTO_PRESERVE_ORDER: Azure, AWS Sagemaker/Bedrock, Private Deployments
    
</dd>
</dl>

<dl>
<dd>

**connectors:** `typing.Optional[typing.List[ChatConnector]]` 

Accepts `{"id": "web-search"}`, and/or the `"id"` for a custom [connector](https://docs.cohere.com/docs/connectors), if you've [created](https://docs.cohere.com/v1/docs/creating-and-deploying-a-connector) one.

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

**documents:** `typing.Optional[typing.List[ChatDocument]]` 

A list of relevant documents that the model can cite to generate a more accurate reply. Each document is a string-string dictionary.

Example:
```
[
  { "title": "Tall penguins", "text": "Emperor penguins are the tallest." },
  { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica." },
]
```

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

Defaults to `"enabled"`.
Citations are enabled by default for models that support it, but can be turned off by setting `"type": "disabled"`.

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

**stop_sequences:** `typing.Optional[typing.List[str]]` 

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

**tools:** `typing.Optional[typing.List[Tool]]` 

A list of available tools (functions) that the model may suggest invoking before producing a text response.

When `tools` is passed (without `tool_results`), the `text` field in the response will be `""` and the `tool_calls` field in the response will be populated with a list of tool calls that need to be made. If no calls need to be made, the `tool_calls` array will be empty.

Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments
    
</dd>
</dl>

<dl>
<dd>

**tool_results:** `typing.Optional[typing.List[ToolResult]]` 

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

**safety_mode:** `typing.Optional[ChatRequestSafetyMode]` 

Used to select the [safety instruction](https://docs.cohere.com/docs/safety-modes) inserted into the prompt. Defaults to `CONTEXTUAL`.
When `NONE` is specified, the safety instruction will be omitted.

Safety modes are not yet configurable in combination with `tools`, `tool_results` and `documents` parameters.

**Note**: This parameter is only compatible newer Cohere models, starting with [Command R 08-2024](https://docs.cohere.com/docs/command-r#august-2024-release) and [Command R+ 08-2024](https://docs.cohere.com/docs/command-r-plus#august-2024-release).

**Note**: `command-r7b-12-2024` and newer models only support `"CONTEXTUAL"` and `"STRICT"` modes.

Compatible Deployments: Cohere Platform, Azure, AWS Sagemaker/Bedrock, Private Deployments
    
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

<details><summary><code>client.<a href="src/cohere/client.py">generate_stream</a>(...) -> typing.Iterator[bytes]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

<Warning>
This API is marked as "Legacy" and is no longer maintained. Follow the [migration guide](https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat) to start using the Chat with Streaming API.
</Warning>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.generate_stream(
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

**stream:** `typing.Literal` 

When `true`, the response will be a JSON stream of events. Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.

The final event will contain the complete response, and will contain an `is_finished` field set to `true`. The event will also contain a `finish_reason`, which can be one of the following:
- `COMPLETE` - the model sent back a finished reply
- `MAX_TOKENS` - the reply was cut off because the model reached the maximum number of tokens for its context length
- `ERROR` - something went wrong when generating the reply
- `ERROR_TOXIC` - the model generated a reply that was deemed toxic
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

The identifier of the model to generate with. Currently available models are `command` (default), `command-nightly` (experimental), `command-light`, and `command-light-nightly` (experimental).
Smaller, "light" models are faster, while larger models will perform better. [Custom models](https://docs.cohere.com/docs/training-custom-models) can also be supplied with their full ID.
    
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

**end_sequences:** `typing.Optional[typing.List[str]]` — The generated text will be cut at the beginning of the earliest occurrence of an end sequence. The sequence will be excluded from the text.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.List[str]]` — The generated text will be cut at the end of the earliest occurrence of a stop sequence. The sequence will be included the text.
    
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

One of `GENERATION|NONE` to specify how and if the token likelihoods are returned with the response. Defaults to `NONE`.

If `GENERATION` is selected, the token likelihoods will only be provided for generated text.

WARNING: `ALL` is deprecated, and will be removed in a future release.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">generate</a>(...) -> Generation</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

<Warning>
This API is marked as "Legacy" and is no longer maintained. Follow the [migration guide](https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat) to start using the Chat API.
</Warning>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.generate_stream(
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

**stream:** `typing.Literal` 

When `true`, the response will be a JSON stream of events. Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.

The final event will contain the complete response, and will contain an `is_finished` field set to `true`. The event will also contain a `finish_reason`, which can be one of the following:
- `COMPLETE` - the model sent back a finished reply
- `MAX_TOKENS` - the reply was cut off because the model reached the maximum number of tokens for its context length
- `ERROR` - something went wrong when generating the reply
- `ERROR_TOXIC` - the model generated a reply that was deemed toxic
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` 

The identifier of the model to generate with. Currently available models are `command` (default), `command-nightly` (experimental), `command-light`, and `command-light-nightly` (experimental).
Smaller, "light" models are faster, while larger models will perform better. [Custom models](https://docs.cohere.com/docs/training-custom-models) can also be supplied with their full ID.
    
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

**end_sequences:** `typing.Optional[typing.List[str]]` — The generated text will be cut at the beginning of the earliest occurrence of an end sequence. The sequence will be excluded from the text.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.List[str]]` — The generated text will be cut at the end of the earliest occurrence of a stop sequence. The sequence will be included the text.
    
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

One of `GENERATION|NONE` to specify how and if the token likelihoods are returned with the response. Defaults to `NONE`.

If `GENERATION` is selected, the token likelihoods will only be provided for generated text.

WARNING: `ALL` is deprecated, and will be removed in a future release.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">embed</a>(...) -> EmbedResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint returns text and image embeddings. An embedding is a list of floating point numbers that captures semantic information about the content that it represents.

Embeddings can be used to create classifiers as well as empower semantic search. To learn more about embeddings, see the embedding page.

If you want to learn more how to use the embedding model, have a look at the [Semantic Search Guide](https://docs.cohere.com/docs/semantic-search).
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.embed(
    texts=[
        "hello",
        "goodbye"
    ],
    model="embed-v4.0",
    input_type="classification",
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

**texts:** `typing.Optional[typing.List[str]]` — An array of strings for the model to embed. Maximum number of texts per call is `96`.
    
</dd>
</dl>

<dl>
<dd>

**images:** `typing.Optional[typing.List[str]]` 

An array of image data URIs for the model to embed. Maximum number of images per call is `1`.

The image must be a valid [data URI](https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data). The image must be in either `image/jpeg`, `image/png`, `image/webp`, or `image/gif` format and has a maximum size of 5MB.

Images are only supported with Embed v3.0 and newer models.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` — ID of one of the available [Embedding models](https://docs.cohere.com/docs/cohere-embed).
    
</dd>
</dl>

<dl>
<dd>

**input_type:** `typing.Optional[EmbedInputType]` 
    
</dd>
</dl>

<dl>
<dd>

**embedding_types:** `typing.Optional[typing.List[EmbeddingType]]` 

Specifies the types of embeddings you want to get back. Not required and default is None, which returns the Embed Floats response type. Can be one or more of the following types.

* `"float"`: Use this when you want to get back the default float embeddings. Supported with all Embed models.
* `"int8"`: Use this when you want to get back signed int8 embeddings. Supported with Embed v3.0 and newer Embed models.
* `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Supported with Embed v3.0 and newer Embed models.
* `"binary"`: Use this when you want to get back signed binary embeddings. Supported with Embed v3.0 and newer Embed models.
* `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Supported with Embed v3.0 and newer Embed models.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">rerank</a>(...) -> RerankResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.rerank(
    documents=[
        {
            "text": "Carson City is the capital city of the American state of Nevada."
        },
        {
            "text": "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan."
        },
        {
            "text": "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages."
        },
        {
            "text": "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district."
        },
        {
            "text": "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."
        }
    ],
    query="What is the capital of the United States?",
    top_n=3,
    model="rerank-v4.0-pro",
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

**documents:** `typing.List[RerankRequestDocumentsItem]` 

A list of document objects or strings to rerank.
If a document is provided the text fields is required and all other fields will be preserved in the response.

The total max chunks (length of documents * max_chunks_per_doc) must be less than 10000.

We recommend a maximum of 1,000 documents for optimal endpoint performance.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` — The identifier of the model to use, eg `rerank-v3.5`.
    
</dd>
</dl>

<dl>
<dd>

**top_n:** `typing.Optional[int]` — The number of most relevant documents or indices to return, defaults to the length of the documents
    
</dd>
</dl>

<dl>
<dd>

**rank_fields:** `typing.Optional[typing.List[str]]` — If a JSON object is provided, you can specify which keys you would like to have considered for reranking. The model will rerank based on order of the fields passed in (i.e. rank_fields=['title','author','text'] will rerank using the values in title, author, text  sequentially. If the length of title, author, and text exceeds the context length of the model, the chunking will not re-consider earlier fields). If not provided, the model will use the default text field for ranking.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">classify</a>(...) -> ClassifyResponse</code></summary>
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
from cohere import Client, ClassifyExample
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.classify(
    examples=[
        ClassifyExample(
            text="Dermatologists don\'t like her!",
            label="Spam",
        ),
        ClassifyExample(
            text="\'Hello, open to this?\'",
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
            text="\'Re: Follow up from today\'s meeting\'",
            label="Not spam",
        ),
        ClassifyExample(
            text="Pre-read for tomorrow",
            label="Not spam",
        )
    ],
    inputs=[
        "Confirm your email address",
        "hey i need u to send some $"
    ],
    model="YOUR-FINE-TUNED-MODEL-ID",
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

**inputs:** `typing.List[str]` 

A list of up to 96 texts to be classified. Each one must be a non-empty string.
There is, however, no consistent, universal limit to the length a particular input can be. We perform classification on the first `x` tokens of each input, and `x` varies depending on which underlying model is powering classification. The maximum token length for each model is listed in the "max tokens" column [here](https://docs.cohere.com/docs/models).
Note: by default the `truncate` parameter is set to `END`, so tokens exceeding the limit will be automatically dropped. This behavior can be disabled by setting `truncate` to `NONE`, which will result in validation errors for longer texts.
    
</dd>
</dl>

<dl>
<dd>

**examples:** `typing.Optional[typing.List[ClassifyExample]]` 

An array of examples to provide context to the model. Each example is a text string and its associated label/class. Each unique label requires at least 2 examples associated with it; the maximum number of examples is 2500, and each example has a maximum length of 512 tokens. The values should be structured as `{text: "...",label: "..."}`.
Note: [Fine-tuned Models](https://docs.cohere.com/docs/classify-fine-tuning) trained on classification examples don't require the `examples` parameter to be passed in explicitly.
    
</dd>
</dl>

<dl>
<dd>

**model:** `typing.Optional[str]` — ID of a [Fine-tuned](https://docs.cohere.com/v2/docs/classify-starting-the-training) Classify model
    
</dd>
</dl>

<dl>
<dd>

**preset:** `typing.Optional[str]` — The ID of a custom playground preset. You can create presets in the [playground](https://dashboard.cohere.com/playground). If you use a preset, all other parameters become optional, and any included parameters will override the preset's parameters.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">summarize</a>(...) -> SummarizeResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

<Warning>
This API is marked as "Legacy" and is no longer maintained. Follow the [migration guide](https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat) to start using the Chat API.
</Warning>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.summarize(
    text="Ice cream is a sweetened frozen food typically eaten as a snack or dessert. It may be made from milk or cream and is flavoured with a sweetener, either sugar or an alternative, and a spice, such as cocoa or vanilla, or with fruit such as strawberries or peaches. It can also be made by whisking a flavored cream base and liquid nitrogen together. Food coloring is sometimes added, in addition to stabilizers. The mixture is cooled below the freezing point of water and stirred to incorporate air spaces and to prevent detectable ice crystals from forming. The result is a smooth, semi-solid foam that is solid at very low temperatures (below 2 °C or 35 °F). It becomes more malleable as its temperature increases.\n\nThe meaning of the name \"ice cream\" varies from one country to another. In some countries, such as the United States, \"ice cream\" applies only to a specific variety, and most governments regulate the commercial use of the various terms according to the relative quantities of the main ingredients, notably the amount of cream. Products that do not meet the criteria to be called ice cream are sometimes labelled \"frozen dairy dessert\" instead. In other countries, such as Italy and Argentina, one word is used fo\r all variants. Analogues made from dairy alternatives, such as goat\'s or sheep\'s milk, or milk substitutes (e.g., soy, cashew, coconut, almond milk or tofu), are available for those who are lactose intolerant, allergic to dairy protein or vegan.",
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

<details><summary><code>client.<a href="src/cohere/client.py">tokenize</a>(...) -> TokenizeResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

**model:** `str` — The input will be tokenized by the tokenizer that is used by this model.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">detokenize</a>(...) -> DetokenizeResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.detokenize(
    tokens=[
        10002,
        2261,
        2012,
        8,
        2792,
        43
    ],
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

**tokens:** `typing.List[int]` — The list of tokens to be detokenized.
    
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

<details><summary><code>client.<a href="src/cohere/client.py">check_api_key</a>() -> CheckApiKeyResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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
<details><summary><code>client.v2.<a href="src/cohere/v2/client.py">chat_stream</a>(...) -> typing.Iterator[bytes]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generates a text response to a user message. To learn how to use the Chat API and RAG follow our [Text Generation guides](https://docs.cohere.com/v2/docs/chat-api).

Follow the [Migration Guide](https://docs.cohere.com/v2/docs/migrating-v1-to-v2) for instructions on moving from API v1 to API v2.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.v2.chat_stream(
    model="command-a-03-2025",
    messages=[
        {
            "role": "user",
            "content": "Tell me about LLMs"
        }
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

**stream:** `typing.Literal` 

Defaults to `false`.

When `true`, the response will be a SSE stream of events.

Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.
    
</dd>
</dl>

<dl>
<dd>

**model:** `str` — The name of a compatible [Cohere model](https://docs.cohere.com/v2/docs/models).
    
</dd>
</dl>

<dl>
<dd>

**messages:** `ChatMessages` 
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.List[ToolV2]]` 

A list of tools (functions) available to the model. The model response may contain 'tool_calls' to the specified tools.

Learn more in the [Tool Use guide](https://docs.cohere.com/docs/tools).
    
</dd>
</dl>

<dl>
<dd>

**strict_tools:** `typing.Optional[bool]` 

When set to `true`, tool calls in the Assistant message will be forced to follow the tool definition strictly. Learn more in the [Structured Outputs (Tools) guide](https://docs.cohere.com/docs/structured-outputs-json#structured-outputs-tools).

**Note**: The first few requests with a new set of tools will take longer to process.
    
</dd>
</dl>

<dl>
<dd>

**documents:** `typing.Optional[typing.List[V2ChatStreamRequestDocumentsItem]]` — A list of relevant documents that the model can cite to generate a more accurate reply. Each document is either a string or document object with content and metadata.
    
</dd>
</dl>

<dl>
<dd>

**citation_options:** `typing.Optional[CitationOptions]` 
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[ResponseFormatV2]` 
    
</dd>
</dl>

<dl>
<dd>

**safety_mode:** `typing.Optional[V2ChatStreamRequestSafetyMode]` 

Used to select the [safety instruction](https://docs.cohere.com/v2/docs/safety-modes) inserted into the prompt. Defaults to `CONTEXTUAL`.
When `OFF` is specified, the safety instruction will be omitted.

Safety modes are not yet configurable in combination with `tools` and `documents` parameters.

**Note**: This parameter is only compatible newer Cohere models, starting with [Command R 08-2024](https://docs.cohere.com/docs/command-r#august-2024-release) and [Command R+ 08-2024](https://docs.cohere.com/docs/command-r-plus#august-2024-release).

**Note**: `command-r7b-12-2024` and newer models only support `"CONTEXTUAL"` and `"STRICT"` modes.
    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` 

The maximum number of output tokens the model will generate in the response. If not set, `max_tokens` defaults to the model's maximum output token limit. You can find the maximum output token limits for each model in the [model documentation](https://docs.cohere.com/docs/models).

**Note**: Setting a low value may result in incomplete generations. In such cases, the `finish_reason` field in the response will be set to `"MAX_TOKENS"`.

**Note**: If `max_tokens` is set higher than the model's maximum output token limit, the generation will be capped at that model-specific maximum limit.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.List[str]]` — A list of up to 5 strings that the model will use to stop generation. If the model generates a string that matches any of the strings in the list, it will stop generating tokens and return the generated text up to that point not including the stop sequence.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` 

Defaults to `0.3`.

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations, and higher temperatures mean more random generations.

Randomness can be further maximized by increasing the  value of the `p` parameter.
    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.
    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.
Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.
    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.
Used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.
    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 

Ensures that only the top `k` most likely tokens are considered for generation at each step. When `k` is set to `0`, k-sampling is disabled.
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

**logprobs:** `typing.Optional[bool]` — Defaults to `false`. When set to `true`, the log probabilities of the generated tokens will be included in the response.
    
</dd>
</dl>

<dl>
<dd>

**tool_choice:** `typing.Optional[V2ChatStreamRequestToolChoice]` 

Used to control whether or not the model will be forced to use a tool when answering. When `REQUIRED` is specified, the model will be forced to use at least one of the user-defined tools, and the `tools` parameter must be passed in the request.
When `NONE` is specified, the model will be forced **not** to use one of the specified tools, and give a direct response.
If tool_choice isn't specified, then the model is free to choose whether to use the specified tools or not.

**Note**: This parameter is only compatible with models [Command-r7b](https://docs.cohere.com/v2/docs/command-r7b) and newer.
    
</dd>
</dl>

<dl>
<dd>

**thinking:** `typing.Optional[Thinking]` 
    
</dd>
</dl>

<dl>
<dd>

**priority:** `typing.Optional[int]` — Controls how early the request is handled. Lower numbers indicate higher priority (default: 0, the highest). When the system is under load, higher-priority requests are processed first and are the least likely to be dropped.
    
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

<details><summary><code>client.v2.<a href="src/cohere/v2/client.py">chat</a>(...) -> V2ChatResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generates a text response to a user message and streams it down, token by token. To learn how to use the Chat API with streaming follow our [Text Generation guides](https://docs.cohere.com/v2/docs/chat-api).

Follow the [Migration Guide](https://docs.cohere.com/v2/docs/migrating-v1-to-v2) for instructions on moving from API v1 to API v2.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.v2.chat_stream(
    model="command-a-03-2025",
    messages=[
        {
            "role": "user",
            "content": "Tell me about LLMs"
        }
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

**stream:** `typing.Literal` 

Defaults to `false`.

When `true`, the response will be a SSE stream of events.

Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.
    
</dd>
</dl>

<dl>
<dd>

**model:** `str` — The name of a compatible [Cohere model](https://docs.cohere.com/v2/docs/models).
    
</dd>
</dl>

<dl>
<dd>

**messages:** `ChatMessages` 
    
</dd>
</dl>

<dl>
<dd>

**tools:** `typing.Optional[typing.List[ToolV2]]` 

A list of tools (functions) available to the model. The model response may contain 'tool_calls' to the specified tools.

Learn more in the [Tool Use guide](https://docs.cohere.com/docs/tools).
    
</dd>
</dl>

<dl>
<dd>

**strict_tools:** `typing.Optional[bool]` 

When set to `true`, tool calls in the Assistant message will be forced to follow the tool definition strictly. Learn more in the [Structured Outputs (Tools) guide](https://docs.cohere.com/docs/structured-outputs-json#structured-outputs-tools).

**Note**: The first few requests with a new set of tools will take longer to process.
    
</dd>
</dl>

<dl>
<dd>

**documents:** `typing.Optional[typing.List[V2ChatRequestDocumentsItem]]` — A list of relevant documents that the model can cite to generate a more accurate reply. Each document is either a string or document object with content and metadata.
    
</dd>
</dl>

<dl>
<dd>

**citation_options:** `typing.Optional[CitationOptions]` 
    
</dd>
</dl>

<dl>
<dd>

**response_format:** `typing.Optional[ResponseFormatV2]` 
    
</dd>
</dl>

<dl>
<dd>

**safety_mode:** `typing.Optional[V2ChatRequestSafetyMode]` 

Used to select the [safety instruction](https://docs.cohere.com/v2/docs/safety-modes) inserted into the prompt. Defaults to `CONTEXTUAL`.
When `OFF` is specified, the safety instruction will be omitted.

Safety modes are not yet configurable in combination with `tools` and `documents` parameters.

**Note**: This parameter is only compatible newer Cohere models, starting with [Command R 08-2024](https://docs.cohere.com/docs/command-r#august-2024-release) and [Command R+ 08-2024](https://docs.cohere.com/docs/command-r-plus#august-2024-release).

**Note**: `command-r7b-12-2024` and newer models only support `"CONTEXTUAL"` and `"STRICT"` modes.
    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` 

The maximum number of output tokens the model will generate in the response. If not set, `max_tokens` defaults to the model's maximum output token limit. You can find the maximum output token limits for each model in the [model documentation](https://docs.cohere.com/docs/models).

**Note**: Setting a low value may result in incomplete generations. In such cases, the `finish_reason` field in the response will be set to `"MAX_TOKENS"`.

**Note**: If `max_tokens` is set higher than the model's maximum output token limit, the generation will be capped at that model-specific maximum limit.
    
</dd>
</dl>

<dl>
<dd>

**stop_sequences:** `typing.Optional[typing.List[str]]` — A list of up to 5 strings that the model will use to stop generation. If the model generates a string that matches any of the strings in the list, it will stop generating tokens and return the generated text up to that point not including the stop sequence.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` 

Defaults to `0.3`.

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations, and higher temperatures mean more random generations.

Randomness can be further maximized by increasing the  value of the `p` parameter.
    
</dd>
</dl>

<dl>
<dd>

**seed:** `typing.Optional[int]` 

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.
    
</dd>
</dl>

<dl>
<dd>

**frequency_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.
Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.
    
</dd>
</dl>

<dl>
<dd>

**presence_penalty:** `typing.Optional[float]` 

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.
Used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.
    
</dd>
</dl>

<dl>
<dd>

**k:** `typing.Optional[int]` 

Ensures that only the top `k` most likely tokens are considered for generation at each step. When `k` is set to `0`, k-sampling is disabled.
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

**logprobs:** `typing.Optional[bool]` — Defaults to `false`. When set to `true`, the log probabilities of the generated tokens will be included in the response.
    
</dd>
</dl>

<dl>
<dd>

**tool_choice:** `typing.Optional[V2ChatRequestToolChoice]` 

Used to control whether or not the model will be forced to use a tool when answering. When `REQUIRED` is specified, the model will be forced to use at least one of the user-defined tools, and the `tools` parameter must be passed in the request.
When `NONE` is specified, the model will be forced **not** to use one of the specified tools, and give a direct response.
If tool_choice isn't specified, then the model is free to choose whether to use the specified tools or not.

**Note**: This parameter is only compatible with models [Command-r7b](https://docs.cohere.com/v2/docs/command-r7b) and newer.
    
</dd>
</dl>

<dl>
<dd>

**thinking:** `typing.Optional[Thinking]` 
    
</dd>
</dl>

<dl>
<dd>

**priority:** `typing.Optional[int]` — Controls how early the request is handled. Lower numbers indicate higher priority (default: 0, the highest). When the system is under load, higher-priority requests are processed first and are the least likely to be dropped.
    
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

<details><summary><code>client.v2.<a href="src/cohere/v2/client.py">embed</a>(...) -> EmbedByTypeResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This endpoint returns text embeddings. An embedding is a list of floating point numbers that captures semantic information about the text that it represents.

Embeddings can be used to create text classifiers as well as empower semantic search. To learn more about embeddings, see the embedding page.

If you want to learn more how to use the embedding model, have a look at the [Semantic Search Guide](https://docs.cohere.com/docs/semantic-search).
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.v2.embed(
    texts=[
        "hello",
        "goodbye"
    ],
    model="embed-v4.0",
    input_type="classification",
    embedding_types=[
        "float"
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

**model:** `str` — ID of one of the available [Embedding models](https://docs.cohere.com/docs/cohere-embed).
    
</dd>
</dl>

<dl>
<dd>

**input_type:** `EmbedInputType` 
    
</dd>
</dl>

<dl>
<dd>

**texts:** `typing.Optional[typing.List[str]]` — An array of strings for the model to embed. Maximum number of texts per call is `96`.
    
</dd>
</dl>

<dl>
<dd>

**images:** `typing.Optional[typing.List[str]]` 

An array of image data URIs for the model to embed. Maximum number of images per call is `1`.

The image must be a valid [data URI](https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data). The image must be in either `image/jpeg`, `image/png`, `image/webp`, or `image/gif` format and has a maximum size of 5MB.

Image embeddings are supported with Embed v3.0 and newer models.
    
</dd>
</dl>

<dl>
<dd>

**inputs:** `typing.Optional[typing.List[EmbedInput]]` — An array of inputs for the model to embed. Maximum number of inputs per call is `96`. An input can contain a mix of text and image components.
    
</dd>
</dl>

<dl>
<dd>

**max_tokens:** `typing.Optional[int]` — The maximum number of tokens to embed per input. If the input text is longer than this, it will be truncated according to the `truncate` parameter.
    
</dd>
</dl>

<dl>
<dd>

**output_dimension:** `typing.Optional[int]` 

The number of dimensions of the output embedding. This is only available for `embed-v4` and newer models.
Possible values are `256`, `512`, `1024`, and `1536`. The default is `1536`.
    
</dd>
</dl>

<dl>
<dd>

**embedding_types:** `typing.Optional[typing.List[EmbeddingType]]` 

Specifies the types of embeddings you want to get back. Can be one or more of the following types.

* `"float"`: Use this when you want to get back the default float embeddings. Supported with all Embed models.
* `"int8"`: Use this when you want to get back signed int8 embeddings. Supported with Embed v3.0 and newer Embed models.
* `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Supported with Embed v3.0 and newer Embed models.
* `"binary"`: Use this when you want to get back signed binary embeddings. Supported with Embed v3.0 and newer Embed models.
* `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Supported with Embed v3.0 and newer Embed models.
* `"base64"`: Use this when you want to get back base64 embeddings. Supported with Embed v3.0 and newer Embed models.
    
</dd>
</dl>

<dl>
<dd>

**truncate:** `typing.Optional[V2EmbedRequestTruncate]` 

One of `NONE|START|END` to specify how the API will handle inputs longer than the maximum token length.

Passing `START` will discard the start of the input. `END` will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model.

If `NONE` is selected, when the input exceeds the maximum input token length an error will be returned.
    
</dd>
</dl>

<dl>
<dd>

**priority:** `typing.Optional[int]` — Controls how early the request is handled. Lower numbers indicate higher priority (default: 0, the highest). When the system is under load, higher-priority requests are processed first and are the least likely to be dropped.
    
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

<details><summary><code>client.v2.<a href="src/cohere/v2/client.py">rerank</a>(...) -> V2RerankResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.v2.rerank(
    documents=[
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."
    ],
    query="What is the capital of the United States?",
    top_n=3,
    model="rerank-v4.0-pro",
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

**model:** `str` — The identifier of the model to use, eg `rerank-v3.5`.
    
</dd>
</dl>

<dl>
<dd>

**query:** `str` — The search query
    
</dd>
</dl>

<dl>
<dd>

**documents:** `typing.List[str]` 

A list of texts that will be compared to the `query`.
For optimal performance we recommend against sending more than 1,000 documents in a single request.

**Note**: long documents will automatically be truncated to the value of `max_tokens_per_doc`.

**Note**: structured data should be formatted as YAML strings for best performance.
    
</dd>
</dl>

<dl>
<dd>

**top_n:** `typing.Optional[int]` — Limits the number of returned rerank results to the specified value. If not passed, all the rerank results will be returned.
    
</dd>
</dl>

<dl>
<dd>

**max_tokens_per_doc:** `typing.Optional[int]` — Defaults to `4096`. Long documents will be automatically truncated to the specified number of tokens.
    
</dd>
</dl>

<dl>
<dd>

**priority:** `typing.Optional[int]` — Controls how early the request is handled. Lower numbers indicate higher priority (default: 0, the highest). When the system is under load, higher-priority requests are processed first and are the least likely to be dropped.
    
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

## Batches
<details><summary><code>client.batches.<a href="src/cohere/batches/client.py">list</a>(...) -> ListBatchesResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List the batches for the current user
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.batches.list(
    page_size=1,
    page_token="page_token",
    order_by="order_by",
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

**page_size:** `typing.Optional[int]` 

The maximum number of batches to return. The service may return fewer than
this value.
If unspecified, at most 50 batches will be returned.
The maximum value is 1000; values above 1000 will be coerced to 1000.
    
</dd>
</dl>

<dl>
<dd>

**page_token:** `typing.Optional[str]` 

A page token, received from a previous `ListBatches` call.
Provide this to retrieve the subsequent page.
    
</dd>
</dl>

<dl>
<dd>

**order_by:** `typing.Optional[str]` 

Batches can be ordered by creation time or last updated time.
Use `created_at` for creation time or `updated_at` for last updated time.
    
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

<details><summary><code>client.batches.<a href="src/cohere/batches/client.py">create</a>(...) -> CreateBatchResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates and executes a batch from an uploaded dataset of requests
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
from cohere import Client
from cohere.environment import ClientEnvironment
from cohere.batches import Batch

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.batches.create(
    request=Batch(
        name="name",
        input_dataset_id="input_dataset_id",
        model="model",
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

**request:** `Batch` 
    
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

<details><summary><code>client.batches.<a href="src/cohere/batches/client.py">retrieve</a>(...) -> GetBatchResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieves a batch
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.batches.retrieve(
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

**id:** `str` — The batch ID.
    
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

<details><summary><code>client.batches.<a href="src/cohere/batches/client.py">cancel</a>(...) -> CancelBatchResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Cancels an in-progress batch
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.batches.cancel(
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

**id:** `str` — The batch ID.
    
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
<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">list</a>() -> ListEmbedJobResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">create</a>(...) -> CreateEmbedJobResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

**embedding_types:** `typing.Optional[typing.List[EmbeddingType]]` 

Specifies the types of embeddings you want to get back. Not required and default is None, which returns the Embed Floats response type. Can be one or more of the following types.

* `"float"`: Use this when you want to get back the default float embeddings. Valid for all models.
* `"int8"`: Use this when you want to get back signed int8 embeddings. Valid for v3 and newer model versions.
* `"uint8"`: Use this when you want to get back unsigned int8 embeddings. Valid for v3 and newer model versions.
* `"binary"`: Use this when you want to get back signed binary embeddings. Valid for v3 and newer model versions.
* `"ubinary"`: Use this when you want to get back unsigned binary embeddings. Valid for v3 and newer model versions.
    
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

<details><summary><code>client.embed_jobs.<a href="src/cohere/embed_jobs/client.py">get</a>(...) -> EmbedJob</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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
<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">list</a>(...) -> DatasetsListResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment
import datetime

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.datasets.list(
    dataset_type="datasetType",
    before=datetime.datetime.fromisoformat("2024-01-15T09:30:00+00:00"),
    after=datetime.datetime.fromisoformat("2024-01-15T09:30:00+00:00"),
    limit=1.1,
    offset=1.1,
    validation_status="unknown",
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

**dataset_type:** `typing.Optional[str]` — optional filter by dataset type
    
</dd>
</dl>

<dl>
<dd>

**before:** `typing.Optional[datetime.datetime]` — optional filter before a date
    
</dd>
</dl>

<dl>
<dd>

**after:** `typing.Optional[datetime.datetime]` — optional filter after a date
    
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

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">create</a>(...) -> DatasetsCreateResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.datasets.create(
    name="name",
    type="embed-input",
    keep_original_file=True,
    skip_malformed_input=True,
    text_separator="text_separator",
    csv_delimiter="csv_delimiter",
    data="example_data",
    eval_data="example_eval_data",
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

**type:** `DatasetType` — The dataset type, which is used to validate the data. The only valid type is `embed-input` used in conjunction with the Embed Jobs API.
    
</dd>
</dl>

<dl>
<dd>

**data:** `core.File` — The file to upload
    
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

**eval_data:** `typing.Optional[core.File]` — An optional evaluation file to upload
    
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

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">get_usage</a>() -> DatasetsGetUsageResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">get</a>(...) -> DatasetsGetResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.datasets.<a href="src/cohere/datasets/client.py">delete</a>(...) -> typing.Dict[str, typing.Any]</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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
<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">list</a>(...) -> ListConnectorsResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.connectors.list(
    limit=1.1,
    offset=1.1,
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

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">create</a>(...) -> CreateConnectorResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new connector. The connector is tested during registration and will cancel registration when the test is unsuccessful. See ['Creating and Deploying a Connector'](https://docs.cohere.com/v1/docs/creating-and-deploying-a-connector) for more information.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

**excludes:** `typing.Optional[typing.List[str]]` — A list of fields to exclude from the prompt (fields remain in the document).
    
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

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">get</a>(...) -> GetConnectorResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">delete</a>(...) -> DeleteConnectorResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">update</a>(...) -> UpdateConnectorResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

**excludes:** `typing.Optional[typing.List[str]]` — A list of fields to exclude from the prompt (fields remain in the document).
    
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

<details><summary><code>client.connectors.<a href="src/cohere/connectors/client.py">o_auth_authorize</a>(...) -> OAuthAuthorizeResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Authorize the connector with the given ID for the connector oauth app.  See ['Connector Authentication'](https://docs.cohere.com/docs/connector-authentication) for more information.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.connectors.o_auth_authorize(
    id="id",
    after_token_redirect="after_token_redirect",
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
<details><summary><code>client.models.<a href="src/cohere/models/client.py">get</a>(...) -> GetModelResponse</code></summary>
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.models.get(
    model="command-a-03-2025",
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

<details><summary><code>client.models.<a href="src/cohere/models/client.py">list</a>(...) -> ListModelsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a list of models available for use.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.models.list(
    page_size=1.1,
    page_token="page_token",
    endpoint="chat",
    default_only=True,
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
<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">list_finetuned_models</a>(...) -> ListFinetunedModelsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a list of fine-tuned models that the user has access to.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.finetuning.list_finetuned_models(
    page_size=1,
    page_token="page_token",
    order_by="order_by",
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

**page_size:** `typing.Optional[int]` 

Maximum number of results to be returned by the server. If 0, defaults to
50.
    
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

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">create_finetuned_model</a>(...) -> CreateFinetunedModelResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new fine-tuned model. The model will be trained on the dataset specified in the request body. The training process may take some time, and the model will be available once the training is complete.
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
from cohere import Client
from cohere.environment import ClientEnvironment
from cohere.finetuning.finetuning import FinetunedModel, Settings, BaseModel

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.finetuning.create_finetuned_model(
    request=FinetunedModel(
        name="name",
        settings=Settings(
            base_model=BaseModel(
                base_type="BASE_TYPE_UNSPECIFIED",
            ),
            dataset_id="dataset_id",
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

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">get_finetuned_model</a>(...) -> GetFinetunedModelResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Retrieve a fine-tuned model by its ID.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">delete_finetuned_model</a>(...) -> DeleteFinetunedModelResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deletes a fine-tuned model. The model will be removed from the system and will no longer be available for use.
This operation is irreversible.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">update_finetuned_model</a>(...) -> UpdateFinetunedModelResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Updates the fine-tuned model with the given ID. The model will be updated with the new settings and name provided in the request body.
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
from cohere import Client
from cohere.environment import ClientEnvironment
from cohere.finetuning.finetuning import Settings, BaseModel

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
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

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">list_events</a>(...) -> ListEventsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a list of events that occurred during the life-cycle of the fine-tuned model.
The events are ordered by creation time, with the most recent event first.
The list can be paginated using `page_size` and `page_token` parameters.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.finetuning.list_events(
    finetuned_model_id="finetuned_model_id",
    page_size=1,
    page_token="page_token",
    order_by="order_by",
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

**page_size:** `typing.Optional[int]` 

Maximum number of results to be returned by the server. If 0, defaults to
50.
    
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

<details><summary><code>client.finetuning.<a href="src/cohere/finetuning/client.py">list_training_step_metrics</a>(...) -> ListTrainingStepMetricsResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a list of metrics measured during the training of a fine-tuned model.
The metrics are ordered by step number, with the most recent step first.
The list can be paginated using `page_size` and `page_token` parameters.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.finetuning.list_training_step_metrics(
    finetuned_model_id="finetuned_model_id",
    page_size=1,
    page_token="page_token",
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

**page_size:** `typing.Optional[int]` 

Maximum number of results to be returned by the server. If 0, defaults to
50.
    
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

## Audio Transcriptions
<details><summary><code>client.audio.transcriptions.<a href="src/cohere/audio/transcriptions/client.py">create</a>(...) -> AudioTranscriptionsCreateResponse</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Transcribe an audio file.
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
from cohere import Client
from cohere.environment import ClientEnvironment

client = Client(
    token="<token>",
    environment=ClientEnvironment.PRODUCTION,
)

client.audio.transcriptions.create(
    file="example_file",
    model="model",
    language="language",
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

**model:** `str` — ID of the model to use.
    
</dd>
</dl>

<dl>
<dd>

**language:** `str` — The language of the input audio, supplied in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) format.
    
</dd>
</dl>

<dl>
<dd>

**file:** `core.File` — The audio file object to transcribe. Supported file extensions are flac, mp3, mpeg, mpga, ogg, and wav.
    
</dd>
</dl>

<dl>
<dd>

**temperature:** `typing.Optional[float]` — The sampling temperature, between 0 and 1. Higher values like 0.8 make the output more random, while lower values like 0.2 make it more focused and deterministic.
    
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

