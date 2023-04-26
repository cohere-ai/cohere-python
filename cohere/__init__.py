from cohere.client import Client
from cohere.client_async import AsyncClient
from cohere.error import CohereAPIError, CohereConnectionError, CohereError

COHERE_API_URL = "https://api.cohere.ai"
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

API_VERSION = "1"
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

CHECK_API_KEY_URL = "check-api-key"
TOKENIZE_URL = "tokenize"
DETOKENIZE_URL = "detokenize"

CLUSTER_JOBS_URL = "cluster-jobs"
BULK_EMBED_JOBS_URL = "embed-jobs"
