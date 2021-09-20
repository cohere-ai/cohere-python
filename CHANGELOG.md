# Changelog

## 0.0.13 - 2021-08-31
* [#26](https://github.com/cohere-ai/cohere-python/pull/26) Add Request Source

## 0.0.11 - 2021-07-22
* [#24](https://github.com/cohere-ai/cohere-python/pull/24) SDK QoL Updates
	* Change from `CohereClient` to be `Client` –– the `CohereClient` will be completely deprecated in the future
	* Have a more human-friendly output when printing Cohere response objects directly

## 0.0.10 - 2021-07-20
* [#23](https://github.com/cohere-ai/cohere-python/pull/23) Add `token_log_likelihoods` to the Choose Best endpoint
* [#21](https://github.com/cohere-ai/cohere-python/pull/21) Change from `BestChoices.likelihoods` to `BestChoices.scores`
* [#19](https://github.com/cohere-ai/cohere-python/pull/19) Make some response objects be iterable


## 0.0.9 - 2021-06-14
* [#18](https://github.com/cohere-ai/cohere-python/pull/18) API Updates - Generate Endpoint
	* Add Frequency Penalty, Presence Penalty, Stop Sequences, and Return Likelihoods for Generate