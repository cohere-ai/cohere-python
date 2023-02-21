import os


def get_api_key() -> str:
    api_key = os.getenv('CO_API_KEY')
    assert api_key, 'CO_API_KEY environment variable not set'
    return api_key


def in_ci() -> bool:
    ci = os.getenv('CI')
    return ci == 'true'
