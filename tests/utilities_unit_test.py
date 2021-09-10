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


from typing import List

import attr
import numpy as np
import pytest

from src.utilities import (
    FromDictMixin,
    cosd,
    sind,
    tand,
    wrap_180,
    wrap_360,
    is_default,
    float_attrib,
    model_attrib,
    iter_validator,
    attrs_array_converter,
)


@attr.s(auto_attribs=True)
class TestClass(FromDictMixin):
    x: int = attr.ib(default=1)
    y: float = float_attrib(default=2.1)
    z: List[str] = attr.ib(default=["empty"], validator=iter_validator(list, str))
    model: str = model_attrib(default="test_class")


@attr.s(auto_attribs=True)
class TestClassArray(FromDictMixin):
    arr: np.ndarray = attr.ib(  # type: ignore
        default=[1, 2], converter=attrs_array_converter, on_setattr=attr.setters.convert  # type: ignore
    )


def test_cosd():
    assert pytest.approx(cosd(0.0)) == 1.0
    assert pytest.approx(cosd(90.0)) == 0.0
    assert pytest.approx(cosd(180.0)) == -1.0
    assert pytest.approx(cosd(270.0)) == 0.0


def test_sind():
    assert pytest.approx(sind(0.0)) == 0.0
    assert pytest.approx(sind(90.0)) == 1.0
    assert pytest.approx(sind(180.0)) == 0.0
    assert pytest.approx(sind(270.0)) == -1.0


def test_tand():
    assert pytest.approx(tand(0.0)) == 0.0
    assert pytest.approx(tand(45.0)) == 1.0
    assert pytest.approx(tand(135.0)) == -1.0
    assert pytest.approx(tand(180.0)) == 0.0
    assert pytest.approx(tand(225.0)) == 1.0
    assert pytest.approx(tand(315.0)) == -1.0


def test_wrap_180():
    assert wrap_180(-180.0) == 180.0
    assert wrap_180(180.0) == 180.0
    assert wrap_180(-181.0) == 179.0
    assert wrap_180(-179.0) == -179.0
    assert wrap_180(179.0) == 179.0
    assert wrap_180(181.0) == -179.0


def test_wrap_360():
    assert wrap_360(0.0) == 0.0
    assert wrap_360(360.0) == 0.0
    assert wrap_360(-1.0) == 359.0
    assert wrap_360(1.0) == 1.0
    assert wrap_360(359.0) == 359.0
    assert wrap_360(361.0) == 1.0


def test_FromDictMixin_defaults():
    inputs = {}
    cls = TestClass.from_dict(inputs)
    assert cls == TestClass()


def test_FromDictMixin_custom():
    # Test custom inputs
    inputs = dict(x=3, y=3.2, z=["one", "two"], arr=[3, 4, 5.5])
    cls = TestClass.from_dict(inputs)
    assert inputs["x"] == cls.x
    assert inputs["y"] == cls.y
    assert inputs["z"] == cls.z
    np.testing.assert_array_equal(inputs["arr"], np.array(inputs["arr"]))


def test_is_default():
    with pytest.raises(ValueError):
        TestClass(model="real deal")


def test_iter_validator():
    # Check wrong member type
    with pytest.raises(TypeError):
        TestClass(z=[4.3, 1])

    # Check mixed member types
    with pytest.raises(TypeError):
        TestClass(z=[4.3, "1"])

    # Check wrong iterable type
    with pytest.raises(TypeError):
        TestClass(z=("a", "b"))


def test_attrs_array_converter():
    test_list = [[1, 2, 3], [4.5, 6.3, 2.2]]
    test_arr = np.array(test_list)

    # Test conversion on initialization
    cls = TestClassArray(arr=test_list)
    np.testing.assert_array_equal(test_arr, cls.arr)

    # Test converstion on reset
    cls = TestClassArray()
    cls.arr = test_list
    np.testing.assert_array_equal(test_arr, cls.arr)


def test_model_attrib():
    with pytest.raises(ValueError):
        TestClass(model="real deal")

    cls = TestClass()
    with pytest.raises(attr.exceptions.FrozenAttributeError):
        cls.model = "real deal"


def test_float_attrib():
    with pytest.raises(ValueError):
        TestClass(y="fail")

    cls = TestClass(y=1)
    assert cls.y == 1.0

    cls.y = "1"
    assert isinstance(cls.y, float)
    assert cls.y == 1.0
