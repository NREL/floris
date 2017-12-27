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
from floris.Coordinate import Coordinate

class CoordinateTest():
    def __init__(self):
        self.x, self.y = self.build_input()

    def build_input(self):
        return 1, 1

# tests

def test_instantiation_with_xy():
    """
    object should be instatiated with x and y
    """
    test_class = CoordinateTest()
    assert Coordinate(test_class.x, test_class.y) != None

def test_rotation():
    """
    Coordinate at 1,1 rotated 90 degrees should result in 1,-1
    """
    test_class = CoordinateTest()
    coordinate = Coordinate(test_class.x, test_class.y)
    coordinate.rotate_z(np.pi/2.0)
    assert pytest.approx(coordinate.xprime) == -1.0 and pytest.approx(coordinate.yprime) == 1.0
