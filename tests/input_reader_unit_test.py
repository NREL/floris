# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation
 

import pytest
from floris.simulation import InputReader


@pytest.fixture
def input_reader_fixture():
    return InputReader()

def test_cast_to_type(input_reader_fixture, sample_inputs_fixture):
    tests = [
        ("1", int, None),
        ("1.0", float, None),
        ("a", str, None),
        # ("{\"a\": 1}", dict, None),
        ("1", list, None)
    ]
    input_reader_fixture.validate_wake(sample_inputs_fixture.floris["wake"])

    # for test in tests:
    #     value = test[0]
    #     typecast = test[1]
    #     expected_error = test[2]
    #     cast_value = input_reader_fixture._cast_to_type(typecast, value)
    #     assert type(cast_value) == typecast

def test_validate_dict(input_reader_fixture):
    """
    This test checks that the function correctly casts a dictionary to the
    mapped type. The "list" type cannot be well tested because `list()` must
    take an iterable as an argument, like list( (1,2,) ). The "dictionary" type
    is also not well tested because traversing through the subdictionary and
    requiring those types to be specified is a bit of a pain.
    Setting "type": "turbine", gets the input past the check for types
    of acceptable kind to FLORIS.
    """
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "list": list,
        "dictionary": dict
    }

    passing_dict = {
        "type": "turbine",
        "name": "test",
        "properties": {
            "string": "1.0",
            "integer": 1,
            "float": 1.0,
            "list": [1.0, 2, 3.0],
            "dictionary": {
                "A": "key",
                "B": 2
            }
        }
    }

    result_dict = input_reader_fixture._validate_dict(passing_dict, type_map)
    assert result_dict == passing_dict

    bad_dict = {
        "type": "turbine",
        "name": "test",
        "properties": {
            "string": 1.0,
            "integer": 1.0,
            "float": 1,
            "list": [1.0, 2, 3.0],
            "dictionary": {
                "A": "key",
                "B": 2
            }
        }
    }

    result_dict = input_reader_fixture._validate_dict(bad_dict, type_map)
    assert result_dict == passing_dict

    bad_dict = {
        "type": "turbine",
        "name": "test",
        "properties": {
            "string": 1.0,
            "integer": 1.0,
            # "float": 1,
            "list": [1.0, 2, 3.0],
            "dictionary": {
                "A": "key",
                "B": 2
            }
        }
    }

    try:
        result_dict = input_reader_fixture._validate_dict(bad_dict, type_map)
    except KeyError:
        pass
    else:
        # If this test case is here, the function is not properly checking
        # that all of the required key-values exist.
        assert result_dict != passing_dict
