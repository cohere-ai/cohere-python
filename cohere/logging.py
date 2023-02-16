import logging

dummy_logger = logging.getLogger("_null_logger")
dummy_logger.addHandler(logging.NullHandler())
dummy_logger.propagate = False

# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class ColoredLogFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: red + format + reset,
        logging.ERROR: bold_red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("cohere-python-sdk")

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(ColoredLogFormatter())

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


def log_off():
    logger.setLevel(logging.CRITICAL)
