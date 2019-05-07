"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np
import pytest
from floris.utilities import Vec3


class Vec3Test():
    def __init__(self):
        self.x, self.y, self.z = self.build_input()
        self.list = [self.x, self.y, self.z]

    def build_input(self):
        return 1, 2, 3


def test_instantiation_with_args():
    """
    The class should initialize with three positional arguments.
    """
    test_class = Vec3Test()

    vec3 = Vec3(test_class.x, test_class.y, test_class.z)
    assert vec3 is not None and \
        vec3.x1 == test_class.x and \
        vec3.x2 == test_class.y and \
        vec3.x3 == test_class.z


def test_instantiation_with_list():
    """
    The class should initialize with a list of length 3.
    """
    test_class = Vec3Test()

    vec3 = Vec3(test_class.list)
    assert vec3 is not None and \
        vec3.x1 == test_class.x and \
        vec3.x2 == test_class.y and \
        vec3.x3 == test_class.z


def test_rotation_on_origin():
    """
    The class should rotate by 180 on the 3rd (z) axis at the origin like so:
        < 1, 2, 3 > becomes < -1, -2, ,3 >
    """
    test_class = Vec3Test()

    baseline = Vec3(-1, -2, 3)
    vec3 = Vec3(test_class.x, test_class.y, test_class.z)
    vec3.rotate_on_x3(180)
    assert \
        vec3.x1prime == pytest.approx(baseline.x1) and \
        vec3.x2prime == pytest.approx(baseline.x2) and \
        vec3.x3prime == pytest.approx(baseline.x3)


def test_rotation_off_origin():
    """    
    The class should rotate by 180 on the 3rd (z) axis about center of rotation at <0, 10, 0> like so:
        < 1, 2, 3 > becomes < -1, -2, ,3 >
    """
    test_class = Vec3Test()

    baseline = Vec3(5, 4, 3)
    center_of_rotation = Vec3(3, 3, 0)
    vec3 = Vec3(test_class.x, test_class.y, test_class.z)
    vec3.rotate_on_x3(180, center_of_rotation)
    assert \
        vec3.x1prime == pytest.approx(baseline.x1) and \
        vec3.x2prime == pytest.approx(baseline.x2) and \
        vec3.x3prime == pytest.approx(baseline.x3)
