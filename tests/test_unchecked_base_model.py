import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cohere.core.unchecked_base_model import construct_type


def test_construct_type_bare_dict_no_unpack_error():
    # bare dict (unparameterized) must not raise ValueError on get_args unpacking
    result = construct_type(type_=dict, object_={"key": "value"})
    assert result == {"key": "value"}
