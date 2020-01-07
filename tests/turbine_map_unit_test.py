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
from .sample_inputs import SampleInputs
from floris.utilities import Vec3
from floris.simulation import Turbine, TurbineMap


class TurbineMapTest():
    def __init__(self):
        self.sample_inputs = SampleInputs()
        self.coordinates = [
            [0.0, 10.0],   # layout x
            [10.0, 20.0]  # layout y
        ]
        self.turbines = [
            Turbine(self.sample_inputs.turbine),
            Turbine(self.sample_inputs.turbine)
        ]
        self.instance = self._build_instance()

    def _build_instance(self):
        return TurbineMap(self.coordinates[0], self.coordinates[1], self.turbines)


def test_instantiation():
    """
    The class should initialize with the standard inputs
    """
    test_class = TurbineMapTest()
    assert test_class.instance is not None


def test_turbines():
    """
    The class should return a dict_items containing all items
    """
    test_class = TurbineMapTest()
    baseline_turbines = test_class.turbines
    test_turbines = test_class.instance.turbines
    for (test, baseline) in zip(test_turbines, baseline_turbines):
        assert test == baseline


def test_coordinates():
    """
    The class should return a dict_items containing all items
    """
    test_class = TurbineMapTest()
    hub_height = test_class.turbines[0].hub_height
    coordinates = [
        [test_class.coordinates[0][0], test_class.coordinates[1][0], hub_height],
        [test_class.coordinates[0][1], test_class.coordinates[1][1], hub_height]
    ]
    baseline_coordinates = [Vec3(c) for c in coordinates]
    test_coordinates = test_class.instance.coords
    for (test, baseline) in zip(test_coordinates, baseline_coordinates):
        assert test == baseline


def test_rotated():
    """
    The class should rotate a turbine when given an angle and center of rotation
    The resulting map should contain turbines at (0, 0) and (-100, 0) when the
    sample map is rotated by pi about (0, 0).
    """
    test_class = TurbineMapTest()
    rotated_map = test_class.instance.rotated([180, 180], Vec3(0.0, 0.0, 0.0))
    baseline_coordinates = [
        Vec3(0.0, -10.0, 90.0),
        Vec3(-10.0, -20.0, 90.0)
    ]
    for i, coordinate in enumerate(rotated_map.coords):
        assert pytest.approx(coordinate.x1) == baseline_coordinates[i].x1
        assert pytest.approx(coordinate.x2) == baseline_coordinates[i].x2
        assert pytest.approx(coordinate.x3) == baseline_coordinates[i].x3


def test_sorted_in_x_as_list():
    """
    The class should sort its Turbines in ascending order based on the 
    x-component of their associated Vec3. The returned object
    should be [(Vec3, Turbine)].
    The resulting list should be ordered as [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
    when the sample data is sorted.
    """
    test_class = TurbineMapTest()
    sorted_map = test_class.instance.sorted_in_x_as_list()
    baseline_coordinates = [
        Vec3(0.0, 10.0, 90.0),
        Vec3(10.0, 20.0, 90.0)
    ]
    for i, element in enumerate(sorted_map):
        coordinate = element[0]
        print(coordinate.x1, baseline_coordinates[i].x1)
        print(coordinate.x2, baseline_coordinates[i].x2)
        print(coordinate.x3, baseline_coordinates[i].x3)
        assert pytest.approx(coordinate.x1) == baseline_coordinates[i].x1
        assert pytest.approx(coordinate.x2) == baseline_coordinates[i].x2
        assert pytest.approx(coordinate.x3) == baseline_coordinates[i].x3
        