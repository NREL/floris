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
from floris.utilities import Vec3


def test_instantiation_with_list():
    """
    The class should initialize with a list of length 3.
    The class should raise an exception if the length of
    points is not 3.
    """
    vec3 = Vec3([1, 2, 3])
    assert vec3.x1 == 1.0
    assert vec3.x2 == 2.0
    assert vec3.x3 == 3.0

    with pytest.raises(Exception):
        vec3 = Vec3([1, 2, 3, 4])

    with pytest.raises(Exception):
        vec3 = Vec3([1, 2])


def test_add(vec3_fixture):
    """
    The overloaded operator should accept a scalar value and apply it to
    all components.
    It should also accept a Vec3 value and perform an element-wise operation.
    """
    scalar = vec3_fixture + 1
    assert scalar.x1 == vec3_fixture.x1 + 1
    assert scalar.x2 == vec3_fixture.x2 + 1
    assert scalar.x3 == vec3_fixture.x3 + 1

    vector = vec3_fixture + Vec3([2, 3, 4])
    assert vector.x1 == vec3_fixture.x1 + 2
    assert vector.x2 == vec3_fixture.x2 + 3
    assert vector.x3 == vec3_fixture.x3 + 4


def test_subtract(vec3_fixture):
    """
    The overloaded operator should accept a scalar value and apply it to
    all components.
    It should also accept a Vec3 value and perform an element-wise operation.
    """
    scalar = vec3_fixture - 1
    assert scalar.x1 == vec3_fixture.x1 - 1
    assert scalar.x2 == vec3_fixture.x2 - 1
    assert scalar.x3 == vec3_fixture.x3 - 1

    vector = vec3_fixture - Vec3([2, 3, 4])
    assert vector.x1 == vec3_fixture.x1 - 2
    assert vector.x2 == vec3_fixture.x2 - 3
    assert vector.x3 == vec3_fixture.x3 - 4


def test_multiply(vec3_fixture):
    """
    The overloaded operator should accept a scalar value and apply it to
    all components.
    It should also accept a Vec3 value and perform an element-wise operation.
    """
    scalar = vec3_fixture * 10
    assert scalar.x1 == vec3_fixture.x1 * 10
    assert scalar.x2 == vec3_fixture.x2 * 10
    assert scalar.x3 == vec3_fixture.x3 * 10

    vector = vec3_fixture * Vec3([2, 3, 4])
    assert vector.x1 == vec3_fixture.x1 * 2
    assert vector.x2 == vec3_fixture.x2 * 3
    assert vector.x3 == vec3_fixture.x3 * 4


def test_divide(vec3_fixture):
    """
    The overloaded operator should accept a scalar value and apply it to
    all components.
    It should also accept a Vec3 value and perform an element-wise operation.
    """
    scalar = vec3_fixture / 10.0
    np.testing.assert_allclose(scalar.x1, vec3_fixture.x1 / 10.0)
    np.testing.assert_allclose(scalar.x2, vec3_fixture.x2 / 10.0)
    np.testing.assert_allclose(scalar.x3, vec3_fixture.x3 / 10.0)

    vector = vec3_fixture / Vec3([10, 100, 1000])
    np.testing.assert_allclose(vector.x1, vec3_fixture.x1 / 10.0)
    np.testing.assert_allclose(vector.x2, vec3_fixture.x2 / 100.0)
    np.testing.assert_allclose(vector.x3, vec3_fixture.x3 / 1000.0)


def test_equality(vec3_fixture):
    """
    The overloaded equality operator should compare each component to the
    same components of the right-hand-side value.
    """
    rhs = Vec3([vec3_fixture.x1, vec3_fixture.x2, vec3_fixture.x3])
    assert vec3_fixture == rhs

    rhs = Vec3([vec3_fixture.x1 + 1, vec3_fixture.x2, vec3_fixture.x3])
    assert vec3_fixture != rhs

    rhs = Vec3([vec3_fixture.x1, vec3_fixture.x2 + 1, vec3_fixture.x3])
    assert vec3_fixture != rhs

    rhs = Vec3([vec3_fixture.x1, vec3_fixture.x2, vec3_fixture.x3 + 1])
    assert vec3_fixture != rhs


def test_elements_property(vec3_fixture):
    """Ensure that the x1, x2, and x3 elements match the expected values.
    """
    x1, x2, x3 = vec3_fixture.elements
    assert 4.0 == x1
    assert 4.0 == x2
    assert 0.0 == x3
