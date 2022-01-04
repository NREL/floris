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

import attrs
from attrs import define, field

from floris.type_dec import (
    FromDictMixin,
    model_attrib,
    iter_validator,
    floris_array_converter
)

@define
class ClassTest(FromDictMixin):
    x: int = field(converter=int)
    y: float = field(converter=float, default=2.1)
    z: List[str] = field(default=["empty"], validator=iter_validator(list, str))
    model: str = model_attrib(default="test_class")


@define
class ArrayTestClass(FromDictMixin):
    arr: np.ndarray = field(  # type: ignore
        default=[1, 2], converter=floris_array_converter, on_setattr=attrs.setters.convert  # type: ignore
    )


def test_FromDictMixin_defaults():
    inputs = {"x": 1}
    cls = ClassTest.from_dict(inputs)
    defaults = {a.name: a.default for a in ClassTest.__attrs_attrs__ if a.default}
    assert cls.y == defaults["y"]
    assert cls.z == defaults["z"]
    assert cls.model == defaults["model"]


def test_FromDictMixin_custom():
    # Test custom inputs
    inputs = dict(x=3, y=3.2, z=["one", "two"])
    cls = ClassTest.from_dict(inputs)
    assert inputs["x"] == cls.x
    assert inputs["y"] == cls.y
    assert inputs["z"] == cls.z

    # Test custom inputs and validate that extra parameters are not mapped
    inputs = dict(x=3, y=3.2, z=["one", "two"], arr=[3, 4, 5.5])
    cls = ClassTest.from_dict(inputs)
    assert inputs["x"] == cls.x
    assert inputs["y"] == cls.y
    assert inputs["z"] == cls.z
    assert not hasattr(cls, "arr")

    # Test that missing required inputs raises an error
    inputs = {}
    with pytest.raises(AttributeError):
        cls = ClassTest.from_dict(inputs)


def test_is_default():
    with pytest.raises(ValueError):
        ClassTest(x=1, model="real deal")


def test_iter_validator():
    # Check wrong member type
    with pytest.raises(TypeError):
        ClassTest(x=1, z=[4.3, 1])

    # Check mixed member types
    with pytest.raises(TypeError):
        ClassTest(x=1, z=[4.3, "1"])

    # Check wrong iterable type
    with pytest.raises(TypeError):
        ClassTest(x=1, z=("a", "b"))


def test_attrs_array_converter():
    test_list = [[1, 2, 3], [4.5, 6.3, 2.2]]
    test_arr = np.array(test_list)

    testtol = 1e-6

    # Test conversion on initialization
    cls = ArrayTestClass(arr=test_list)
    np.testing.assert_allclose(test_arr, cls.arr, atol=testtol)

    # Test converstion on reset
    cls = ArrayTestClass()
    cls.arr = test_list
    np.testing.assert_allclose(test_arr, cls.arr, atol=testtol)


def test_model_attrib():
    with pytest.raises(ValueError):
        ClassTest(x=1, model="real deal")

    cls = ClassTest(x=1)
    with pytest.raises(attrs.exceptions.FrozenAttributeError):
        cls.model = "real deal"


def test_float_attrib():
    with pytest.raises(ValueError):
        ClassTest(x=1, y="fail")

    cls = ClassTest(x=1, y=1)
    assert cls.y == 1.0

    cls.y = "1"
    assert isinstance(cls.y, float)
    assert cls.y == 1.0