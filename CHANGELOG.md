# Changelog

## 4.6.0
- [#238](https://github.com/cohere-ai/cohere-python/pull/238)
  - Add `is_finished` to each element of the streaming chat response
  - Add `conversation_id`, `response_id`, `finish_reason`, `chatlog`, `preamble` and `prompt` to the streaming chat response
  - Fix chat streaming index

## 4.5.1
- [#225](https://github.com/cohere-ai/cohere-python/pull/224)
  - Remove support for the co.chat parameter `chatlog_override` and add deprecation warning

## 4.5.0
- [#229](https://github.com/cohere-ai/cohere-python/pull/229)
  - Add `return_exceptions` parameter to Client's `batch_*` methods, mirroring AsyncClient

## 4.4.2
- [#230](https://github.com/cohere-ai/cohere-python/pull/230)
  - Add SDK level validation for classify params

## 4.4.1
- [#224](https://github.com/cohere-ai/cohere-python/pull/224)
  - Update co.chat parameter `chat_history`

## 4.4.0
- [#223](https://github.com/cohere-ai/cohere-python/pull/223)
  - Remove deprecated co.chat parameter `reply`

## 4.3.1
- [#220](https://github.com/cohere-ai/cohere-python/pull/220)
  - Update chat params
    - Add support for `chat_history`

## 4.3.0

- [#210](https://github.com/cohere-ai/cohere-python/pull/210)
  - Update embed with compressed embeddings
    - `compress`
    - `compression_codebook`
- [#211](https://github.com/cohere-ai/cohere-python/pull/211)
  - Add co.codebook endpoint for compressed embeddings

## 4.2.1

- [#214](https://github.com/cohere-ai/cohere-python/pull/214)
  - Add support for co.Chat parameter:
    - `return_preamble`

## 4.2.0

- [#212](https://github.com/cohere-ai/cohere-python/pull/212)
  - Deprecate co.chat params
    - `session_id`
    - `persona_name`
    - `persona_prompt`
  - Add deprecation warning for Chat attribute
    - Use `text` instead of `reply`
  - Add support for `generation_id`
- [#206](https://github.com/cohere-ai/cohere-python/pull/206)
  - Update cluster endpoint to use UMAP+HDBSCAN
  - Remove threshold and add n_neighbors and is_deterministic as params

## 4.1.6

- [#205](https://github.com/cohere-ai/cohere-python/pull/205)
  - Add param max_chunks_per_doc to rerank
  - Enforce model param for rerank

## 4.1.5

- [#208](https://github.com/cohere-ai/cohere-python/pull/208)
  - Fix a missing import for CohereConnectionError

## 4.1.4

- [#204](https://github.com/cohere-ai/cohere-python/pull/204)
  - Add `generate_preference_feedback` for submitting preference-style feedback

## 4.1.3

- [#194](https://github.com/cohere-ai/cohere-python/pull/194)
  - Return the generation ID for chat

## 4.1.2

- [#192](https://github.com/cohere-ai/cohere-python/pull/192)
  - Fix duplicate Generate calls in the sync SDK

## 4.1.1

- [#190](https://github.com/cohere-ai/cohere-python/pull/190)
  - Remove wrong "Embedding" class used for type hinting

## 4.1.0

- [#188](https://github.com/cohere-ai/cohere-python/pull/188)
  - Add `stream` parameter to chat, and relevant return object.
- [#169](https://github.com/cohere-ai/cohere-python/pull/169)
  - Add `stream` parameter to generate, and relevant return object.
  - Add example notebook for streaming.

## 4.0.6

- [#187](https://github.com/cohere-ai/cohere-python/pull/187)
  - Refactor feedback to be generate specific

## 4.0.5

- [#186](https://github.com/cohere-ai/cohere-python/pull/186)
  - Added warnings support for meta response

## 4.0.4

- [#185](https://github.com/cohere-ai/cohere-python/pull/185)
  - Validate API key without API call

## 4.0.3

- [#184](https://github.com/cohere-ai/cohere-python/pull/184)
  - Respect timeout option for sync client

## 4.0.2

- [#183](https://github.com/cohere-ai/cohere-python/pull/183)
  - Better error messages for synchronous client

## 4.0.1

- [#181](https://github.com/cohere-ai/cohere-python/pull/181)
  - Allow Python >=3.11

## 4.0

- [#160](https://github.com/cohere-ai/cohere-python/pull/160)
  - Add AsyncClient
  - Default value of API key from environment variable `CO_API_KEY`.
  - Feedback endpoint moved from CohereObject to Client/AsyncClient.
  - Lazy initialization using futures removed.
  - Generations is now a UserList, and initialized from responses using `from_dict`.
  - Chat objects are initialized using `from_dict`. Optional attributes are now `None` rather than missing.
  - Documentation expanded and built using sphinx.
  - Use Poetry, and format using black and isort, include pre-commit hooks.
  - Removed ability for user to choose API version. This SDK version defaults to v1.
  - Added 'meta' fields to response objects with API version
- [#179](https://github.com/cohere-ai/cohere-python/pull/179)
  - Add support for co.Chat parameters: `temperature`, `max_tokens`, `persona_name`, `persona_prompt`
  - Remove support for co.Chat parameters: `persona`, `preamble_override`
  - Updates the co.Chat `user_name` parameter

## 3.10.0

- [#176](https://github.com/cohere-ai/cohere-python/pull/176)
  - Add failure reason to clustering jobs

## 3.9.1

- [#175](https://github.com/cohere-ai/cohere-python/pull/175)
  - Fix url path for cluster-job get endpoint

## 3.9

- [#168](https://github.com/cohere-ai/cohere-python/pull/168)
  - Add support for co.Rerank parameter:
    - `model`

## 3.8

- [#158](https://github.com/cohere-ai/cohere-python/pull/158)
  - Add support for co.Chat parameters:
    - `preamble_override`
    - `return_prompt`
    - `username`

## 3.7

- [#164](https://github.com/cohere-ai/cohere-python/pull/164)
  - Add clustering methods:
    - `co.create_cluster_job`
    - `co.get_cluster_job`
    - `co.list_cluster_jobs`
    - `co.wait_for_cluster_job`

## 3.6

- [#156](https://github.com/cohere-ai/cohere-python/pull/156)
  - Replace `abstractiveness` param with `extractiveness` in co.Summarize
  - Rename `additional_instruction` param to `additional_command` in co.Summarize

## 3.5

- [#157](https://github.com/cohere-ai/cohere-python/pull/157)
  - Add support for `chatlog_override` parameter for co.Chat

## 3.4

- [#154](https://github.com/cohere-ai/cohere-python/pull/154)
  - Add support for `return_chatlog` parameter for co.Chat

## 3.3

- [#146](https://github.com/cohere-ai/cohere-python/pull/146)
  - Add new experimental `co.rerank` api
- [#150](https://github.com/cohere-ai/cohere-python/pull/150)
  - Add `abstractiveness` param for `co.summarize`

## 3.2

- [#138](https://github.com/cohere-ai/cohere-python/pull/138)
  - Add `model` to the chat endpoint's parameters
- [#145](https://github.com/cohere-ai/cohere-python/pull/145)
  - Add new experimental `co.summarize` API

## 3.1

- [#129](https://github.com/cohere-ai/cohere-python/pull/129)
  - Add support for `end_sequences` param in Generate API

## 3.1

- [#126](https://github.com/cohere-ai/cohere-python/pull/126)
  - Add new `co.detect_language` api

## 3.0

- [#125](https://github.com/cohere-ai/cohere-python/pull/123)
  - Improve the Classify response string representation

## 2.9

- [#120](https://github.com/cohere-ai/cohere-python/pull/120)
  - Remove experimental Extract API from the SDK

## 2.8

- [#112](https://github.com/cohere-ai/cohere-python/pull/112)
  - Add support for `prompt_vars` parameter for co.Generate

## 2.7

- [#110](https://github.com/cohere-ai/cohere-python/pull/110)
  - Classification.confidence is now a float instead of a list

## 2.6

- [#105](https://github.com/cohere-ai/cohere-python/pull/105)
  - Remove deprecated options from classify
- [#104](https://github.com/cohere-ai/cohere-python/pull/104)
  - Remove experimental `moderate` api

## 2.5

- [#96](https://github.com/cohere-ai/cohere-python/pull/96)
  - The default `max_tokens` value is now configured on the backend

## 2.4

- [#102](https://github.com/cohere-ai/cohere-python/pull/102)
  - Generate Parameter now accepts `logit_bias` as a parameter

## 2.2

- [#95](https://github.com/cohere-ai/cohere-python/pull/95)
  - Introduce Detokenize for converting a list of tokens to a string
- [#92](https://github.com/cohere-ai/cohere-python/pull/92)
  - Handle `truncate` parameter for Classify and Generate

## 1.3

- [#71](https://github.com/cohere-ai/cohere-python/pull/71) Sunset Choose Best

## 1.0

- [#38](https://github.com/cohere-ai/cohere-python/pull/38)
  - Handle `truncate` parameter for Embed
- [#36](https://github.com/cohere-ai/cohere-python/pull/36)
  Change generations to return `Generations`, which has as a list of `Generation` \* Each `Generation` has a `text` and `token_likelihoods` field to store generations and token likelihoods respectively
- [#34](https://github.com/cohere-ai/cohere-python/pull/34)
  API Updates and SDK QoL Improvements
  _Add support for multiple generations
  _ Add capability to use a specific API version \* Fully remove `CohereClient`
- [#32](https://github.com/cohere-ai/cohere-python/pull/32)
  Handle different errors more safely

## 0.0.13 - 2021-08-31

- [#26](https://github.com/cohere-ai/cohere-python/pull/26) Add Request Source

## 0.0.11 - 2021-07-22

- [#24](https://github.com/cohere-ai/cohere-python/pull/24) SDK QoL Updates
  - Change from `CohereClient` to be `Client` –– the `CohereClient` will be completely deprecated in the future
  - Have a more human-friendly output when printing Cohere response objects directly

## 0.0.10 - 2021-07-20

- [#23](https://github.com/cohere-ai/cohere-python/pull/23) Add `token_log_likelihoods` to the Choose Best endpoint
- [#21](https://github.com/cohere-ai/cohere-python/pull/21) Change from `BestChoices.likelihoods` to `BestChoices.scores`
- [#19](https://github.com/cohere-ai/cohere-python/pull/19) Make some response objects be iterable

## 0.0.9 - 2021-06-14

- [#18](https://github.com/cohere-ai/cohere-python/pull/18) API Updates - Generate Endpoint
  - Add Frequency Penalty, Presence Penalty, Stop Sequences, and Return Likelihoods for Generate
