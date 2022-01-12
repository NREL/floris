# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import pytest
from typing import List

from attrs import define, field

from floris.type_dec import (
    FromDictMixin,
    iter_validator,
    floris_array_converter,
)

@define
class AttrsDemoClass(FromDictMixin):
    w: int
    x: int = field(converter=int)
    y: float = field(converter=float, default=2.1)
    z: str = field(converter=str, default="z")

    liststr: List[str] = field(
        default=["qwerty", "asdf"],
        validator=iter_validator(list, str)
    )
    array: np.ndarray = field(
        default=[1.0, 2.0],
        converter=floris_array_converter,
        # validator=iter_validator(np.ndarray, floris_float_type)
    )


def test_FromDictMixin_defaults():
    # Test that the defaults set in the class definition are actually used
    inputs = {"w": 0, "x": 1}
    cls = AttrsDemoClass.from_dict(inputs)
    defaults = {a.name: a.default for a in AttrsDemoClass.__attrs_attrs__ if a.default}
    assert cls.y == defaults["y"]
    assert cls.z == defaults["z"]
    np.testing.assert_array_equal(cls.liststr, defaults["liststr"])
    np.testing.assert_array_equal(cls.array, defaults["array"])

    # Test that defaults can be overwritten
    inputs = {"w": 0, "x": 1, "y": 4.5}
    cls = AttrsDemoClass.from_dict(inputs)
    defaults = {a.name: a.default for a in AttrsDemoClass.__attrs_attrs__ if a.default}
    assert cls.y != defaults["y"]


def test_FromDictMixin_custom():

    inputs = {
        "w": 0,
        "x": 1,
        "y": 2.3,
        "z": "asdf",
        "liststr": ["a", "b"],
        "array": np.array([[1,2,3], [4,5,6]])
    }

    # Check that custom inputs are accepted
    AttrsDemoClass.from_dict(inputs)

    # Ensure extraneous inputs are not applied to the class
    inputs2 = {**inputs, "extra": [3, 4, 5.5]}
    with pytest.raises(AttributeError):
        AttrsDemoClass.from_dict(inputs2)

    # Test that missing required inputs raises an error
    inputs = {}
    with pytest.raises(AttributeError):
        AttrsDemoClass.from_dict(inputs)


def test_iter_validator():

    # Check the correct values work
    _ = AttrsDemoClass(w=0, x=1, liststr=["a", "b"])

    # Check wrong member type
    with pytest.raises(TypeError):
        AttrsDemoClass(w=0, x=1, liststr=[4.3, 1])

    # Check mixed member types
    with pytest.raises(TypeError):
        AttrsDemoClass(w=0, x=1, liststr=[4.3, "1"])

    # Check wrong iterable type
    with pytest.raises(TypeError):
        AttrsDemoClass(w=0, x=1, liststr=("a", "b"))


def test_attrs_array_converter():
    array_input = [[1, 2, 3], [4.5, 6.3, 2.2]]
    test_array = np.array(array_input)

    # Test conversion on initialization
    cls = AttrsDemoClass(w=0, x=1, array=array_input)
    np.testing.assert_allclose(test_array, cls.array)

    # Test converstion on reset
    cls.array = array_input
    np.testing.assert_allclose(test_array, cls.array)
