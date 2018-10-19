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
from floris.coordinate import Coordinate


class CoordinateTest():
    def __init__(self):
        self.x, self.y = self.build_input()

    def build_input(self):
        return 1, 1


def test_instantiation_with_xy():
    """
    The class should initialize with the standard inputs
    """
    test_class = CoordinateTest()
    coordinate = Coordinate(test_class.x, test_class.y)
    assert coordinate is not None and \
        coordinate.x == test_class.x and \
        coordinate.y == test_class.y and \
        coordinate.z == 0 and \
        coordinate.xprime == test_class.x and \
        coordinate.yprime == test_class.y and \
        coordinate.zprime == 0


def test_string_format():
    """
    Coordinate should print its coordinates wrapped in parenthesis when cast to string
    """
    test_class = CoordinateTest()
    coordinate = Coordinate(test_class.x, test_class.y)
    assert str(coordinate) == "({}, {})".format(test_class.x, test_class.y)


def test_as_tuple():
    """
    Coordinate return its x and y coordinates as a tuple
    """
    test_class = CoordinateTest()
    coordinate = Coordinate(test_class.x, test_class.y)
    assert coordinate.as_tuple() == tuple([test_class.x, test_class.y])


def test_rotation_on_z():
    """
    Coordinate at 1, 1 rotated 90 degrees around z axis should result in 1,-1
    """
    test_class = CoordinateTest()
    coordinate = Coordinate(test_class.x, test_class.y)
    coordinate.rotate_z(np.pi / 2.0)
    assert pytest.approx(coordinate.xprime) == -1.0 \
        and pytest.approx(coordinate.yprime) == 1.0
