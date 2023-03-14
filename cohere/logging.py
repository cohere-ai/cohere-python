import logging

logger = logging.getLogger("cohere-python-sdk")

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

logger.addHandler(ch)
logger.setLevel(logging.WARNING)
logger.propagate = False


def log_debug():
    logger.setLevel(logging.DEBUG)


def log_info():
    logger.setLevel(logging.INFO)


def log_warning():
    logger.setLevel(logging.WARNING)


def log_error():
    logger.setLevel(logging.ERROR)


def log_critical():
    logger.setLevel(logging.CRITICAL)
