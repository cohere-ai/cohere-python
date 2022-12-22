# Changelog

# 3.1.2

- [#129](https://github.com/cohere-ai/cohere-python/pull/129)
  - Add support for `end_sequences` param in Generate API

# 3.1.0

- [#126](https://github.com/cohere-ai/cohere-python/pull/126)
  - Add new `co.detect_language` api

# 3.0.1

- [#125](https://github.com/cohere-ai/cohere-python/pull/123)
  - Improve the Classify response string representation

# 3.0.0

- [#125](https://github.com/cohere-ai/cohere-python/pull/125)
  - Remove the deprecated "confidences" field from the classify response

## 2.9.0

- [#120](https://github.com/cohere-ai/cohere-python/pull/120)
  - Remove experimental Extract API from the SDK

## 2.8.0

- [#112](https://github.com/cohere-ai/cohere-python/pull/112)
  - Add support for `prompt_vars` parameter for co.Generate

## 2.7.0

- [#110](https://github.com/cohere-ai/cohere-python/pull/110)
  - Classification.confidence is now a float instead of a list

## 2.6.1

- [#105](https://github.com/cohere-ai/cohere-python/pull/105)
  - Remove deprecated options from classify

## 2.6.0

- [#104](https://github.com/cohere-ai/cohere-python/pull/104)
  - Remove experimental `moderate` api

## 2.5.0

- [#96](https://github.com/cohere-ai/cohere-python/pull/96)
  - The default `max_tokens` value is now configured on the backend

## 2.4.2

- [#102](https://github.com/cohere-ai/cohere-python/pull/102)
  - Generate Parameter now accepts `logit_bias` as a parameter

## 2.2.5

- [#95](https://github.com/cohere-ai/cohere-python/pull/95)
  - Introduce Detokenize for converting a list of tokens to a string

## 2.2.4

- [#92](https://github.com/cohere-ai/cohere-python/pull/92)
  - Handle `truncate` parameter for Classify and Generate

## 1.3.6 - 2022-05-05

- [#71](https://github.com/cohere-ai/cohere-python/pull/71) Sunset Choose Best

## 1.0.2 - 2021-11-30

- [#38](https://github.com/cohere-ai/cohere-python/pull/38)
  - Handle `truncate` parameter for Embed

## 1.0.1 - 2021-11-17

- [#36](https://github.com/cohere-ai/cohere-python/pull/36)
  Change generations to return `Generations`, which has as a list of `Generation` \* Each `Generation` has a `text` and `token_likelihoods` field to store generations and token likelihoods respectively
- [#34](https://github.com/cohere-ai/cohere-python/pull/34)
  API Updates and SDK QoL Improvements
  _ Add support for multiple generations
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
