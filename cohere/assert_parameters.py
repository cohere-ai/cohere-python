from cohere.error import CohereError
from typing import Any, Union

def assert_parameter (expected_type: Union[type, tuple], name: str, input: Any, endpoint: str):
    base_error = get_base_error(expected_type, name)
    
    reference_link = f"https://docs.cohere.ai/{endpoint}-reference/"
    error = CohereError(message = f"{base_error} but received a {type(input).__name__}. See {reference_link} for more details.")

    if not isinstance(input, expected_type):
        raise error

def assert_list_parameter (expected_type: Union[type, tuple], name: str, input: Any, endpoint: str):
    if isinstance(input, expected_type):
        return [input]
    
    assert_parameter(list, name, input, endpoint)
    base_error = get_base_error(expected_type, name)
    reference_link = f"https://docs.cohere.ai/{endpoint}-reference/"
    if not all(isinstance(elem, expected_type) for elem in input):
        for elem in input:
            if not isinstance(elem, expected_type):
                unexpected_type = type(elem).__name__
                index = input.index(elem)
        raise CohereError(
            message = f"{base_error} but found a {unexpected_type} at index {index}. See {reference_link} for more details."
        )

    return input

def get_base_error (expected_type: Union[type, tuple], name: str) -> str:
    base_error = ""
    if hasattr(expected_type, '__name__'):
        return f"the {name} parameter is expected to be a list of {expected_type.__name__},"
    else:
        expected_types = expected_type[0].__name__
        for my_type in expected_type[1:]:
            expected_types += " or a " + my_type.__name__ 
        return f"the {name} parameter is expected to be a {expected_types},"

