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
import copy
import pytest
from .sample_inputs import SampleInputs
from floris.coordinate import Coordinate
from floris.turbine import Turbine
from floris.turbine_map import TurbineMap

class TurbineMapTest():
    def __init__(self):
        self.sample_inputs = SampleInputs()
        self.coordinates = [
            Coordinate(0.0, 0.0),
            Coordinate(100.0, 0.0)
        ]
        self.turbine_map_dict = self._build_turbine_map_dict()
        self.instance = self._build_instance()

    def _build_turbine_map_dict(self):
        return {
            self.coordinates[0]: Turbine(self.sample_inputs.turbine),
            self.coordinates[1]: Turbine(self.sample_inputs.turbine)
        }

    def _build_instance(self):
        return TurbineMap(self.turbine_map_dict)


def test_instantiation():
    """
    The class should initialize with the standard inputs
    """
    test_class = TurbineMapTest()
    assert test_class.instance is not None


def test_items():
    """
    The class should return a dict_items containing all items
    """
    test_class = TurbineMapTest()
    items = test_class.instance.items()
    for i, item in enumerate(items):
        assert test_class.coordinates[i] is item[0]
        assert test_class.turbine_map_dict[test_class.coordinates[i]] is item[1]


def test_rotated():
    """
    The class should rotate a turbine when given an angle and center of rotation
    The resulting map should contain turbines at (0, 0) and (-100, 0) when the
    sample map is rotated by pi about (0, 0).
    """
    test_class = TurbineMapTest()
    rotated_map = test_class.instance.rotated(np.pi, Coordinate(0, 0))
    baseline_coordinates = [
        Coordinate(0.0, 0.0),
        Coordinate(-100.0, 0.0)
    ]
    for i, coordinate in enumerate(rotated_map.coords):
        assert pytest.approx(coordinate == baseline_coordinates[i])


def test_sorted_in_x_as_list():
    """
    The class should sort its Turbines in ascending order based on the 
    x-component of their associated Coordinate. The returned object
    should be [(Coordinate, Turbine)].
    The resulting list should be ordered as [(0.0, 0.0), (100.0, 0.0)] when the
    sample data is sorted.
    """
    test_class = TurbineMapTest()
    sorted_map = test_class.instance.sorted_in_x_as_list()
    baseline_coordinates = [
        Coordinate(0.0, 0.0),
        Coordinate(-100.0, 0.0)
    ]
    for i, element in enumerate(sorted_map):
        coordinate = element[0]
        assert pytest.approx(coordinate == baseline_coordinates[i])


def sorted_in_x_as_list(self):
    coords = sorted(self.turbine_map_dict, key=lambda coord: coord.x)
    return [(c, self.turbine_map_dict[c]) for c in coords]
