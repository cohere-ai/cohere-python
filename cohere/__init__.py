from .client import Client
from .error import CohereError

COHERE_API_URL = 'https://api.cohere.ai'
COHERE_VERSION = '2021-11-08'
COHERE_EMBED_BATCH_SIZE = 16
GENERATE_URL = 'generate'
EMBED_URL = 'embed'
CHOOSE_BEST_URL = 'choose-best'
CLASSIFY_URL = 'classify'

CHECK_API_KEY_URL = 'check-api-key'
TOKENIZE_URL = 'tokenize'
