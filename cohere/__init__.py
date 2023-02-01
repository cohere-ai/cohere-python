from .client import Client
from .error import CohereError

COHERE_API_URL = 'https://api.cohere.ai'
COHERE_VERSION = '2022-12-06'
COHERE_EMBED_BATCH_SIZE = 96
CHAT_URL = 'chat'
CLASSIFY_URL = 'classify'
DETECT_LANG_URL = 'detect-language'
EMBED_URL = 'embed'
FEEDBACK_URL = 'feedback'
GENERATE_URL = 'generate'
SUMMARIZE_URL = 'summarize'
RERANK_URL = 'rerank'

CHECK_API_KEY_URL = 'check-api-key'
TOKENIZE_URL = 'tokenize'
DETOKENIZE_URL = 'detokenize'
