from importlib_metadata import version  # use package to support python 3.7

from cohere.client import Client
from cohere.client_async import AsyncClient
from cohere.error import CohereAPIError, CohereConnectionError, CohereError

COHERE_API_URL = "https://api.cohere.ai"
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

API_VERSION = "1"

SDK_VERSION = version("cohere")
COHERE_EMBED_BATCH_SIZE = 96
CHAT_URL = "chat"
CLASSIFY_URL = "classify"
CODEBOOK_URL = "embed-codebook"
DETECT_LANG_URL = "detect-language"
EMBED_URL = "embed"
GENERATE_FEEDBACK_URL = "feedback/generate"
GENERATE_PREFERENCE_FEEDBACK_URL = "feedback/generate/preference"
GENERATE_URL = "generate"
SUMMARIZE_URL = "summarize"
RERANK_URL = "rerank"
DATASET_URL = "dataset"

CHECK_API_KEY_URL = "check-api-key"
TOKENIZE_URL = "tokenize"
DETOKENIZE_URL = "detokenize"
LOGLIKELIHOOD_URL = "loglikelihood"

CLUSTER_JOBS_URL = "cluster-jobs"
EMBED_JOBS_URL = "embed-jobs"
CUSTOM_MODEL_URL = "finetune"

OCI_COHERE_API_URL = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"
OCI_API_TYPE = "oci"
OCI_API_VERSION = "20231130"

OCI_CHAT_URL = "chat"
OCI_CLASSIFY_URL = "classify"
OCI_CODEBOOK_URL = "embed-codebook"
OCI_DETECT_LANG_URL = "detect-language"
OCI_EMBED_URL = "actions/embedText"
OCI_GENERATE_FEEDBACK_URL = "feedback/generate"
OCI_GENERATE_PREFERENCE_FEEDBACK_URL = "feedback/generate/preference"
OCI_GENERATE_URL = "actions/generateText"
OCI_SUMMARIZE_URL = "actions/summarizeText"
OCI_RERANK_URL = "rerank"
OCI_DATASET_URL = "dataset"

OCI_CHECK_API_KEY_URL = "check-api-key"
OCI_TOKENIZE_URL = "tokenize"
OCI_DETOKENIZE_URL = "detokenize"
OCI_LOGLIKELIHOOD_URL = "loglikelihood"

OCI_CLUSTER_JOBS_URL = "cluster-jobs"
OCI_EMBED_JOBS_URL = "embed-jobs"
OCI_CUSTOM_MODEL_URL = "finetune"
