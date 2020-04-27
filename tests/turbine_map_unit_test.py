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
from floris.utilities import Vec3
from floris.simulation import Turbine, TurbineMap


@pytest.fixture
def turbine_map_fixture(sample_inputs_fixture):
    return TurbineMap(
        sample_inputs_fixture.farm["properties"]["layout_x"],
        sample_inputs_fixture.farm["properties"]["layout_y"],
        3 * [Turbine(sample_inputs_fixture.turbine)]
    )

def test_instantiation(turbine_map_fixture):
    """
    The class should initialize with the standard inputs
    """
    assert type(turbine_map_fixture) is TurbineMap

def test_rotated(turbine_map_fixture):
    """
    The class should rotate the location of the turbines when given an angle
    and center of rotation. In this case, the turbines are in a line and
    rotated 180 degrees about the middle turbine. The expected result is that
    the first turbine is located at the last turbine, the second turbine
    does not move, and the last turbine is located at the first turbine.
    """
    first = turbine_map_fixture.coords[0]
    third = turbine_map_fixture.coords[2]
    rotated_map = turbine_map_fixture.rotated(
        [180, 180, 180],
        (first + third) / 2.0
    )
    fixture_coordinates = turbine_map_fixture.coords
    rotated_coordinates = rotated_map.coords
    assert rotated_coordinates[0] == fixture_coordinates[2]
    assert rotated_coordinates[1] == fixture_coordinates[1]
    assert rotated_coordinates[2] == fixture_coordinates[0]

def test_sorted_in_x_as_list(turbine_map_fixture):
    """
    The class should sort its Turbines in ascending order based on the 
    x-component of their associated coordinate (Vec3). The returned object
    should be [(Vec3, Turbine)].
    The resulting list should be ordered as [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
    when the sample data is sorted.
    """
    sorted_list = turbine_map_fixture.sorted_in_x_as_list()

    # Verify that each element of the returned list is a tuple (Vec3, Turbine)
    for coordinate, turbine in sorted_list:
        assert isinstance(coordinate, Vec3)
        assert isinstance(turbine, Turbine)
    
    # Verify that the coordinates are sorted in ascending order in the
    # x direction
    coordinates = [x[0] for x in sorted_list]
    previous = coordinates[0]
    for i, c in enumerate(coordinates):
        if i == 0:
            previous = c
            continue
        assert c.x1 > previous.x1

def test_turbines(turbine_map_fixture):
    """
    The class should return a list containing all turbines
    """
    test_turbines = turbine_map_fixture.turbines
    assert len(test_turbines) == 3
    for t in test_turbines:
        assert isinstance(t, Turbine)

def test_coordinates(turbine_map_fixture):
    """
    The class should return a list containing coordinates
    """
    test_coords = turbine_map_fixture.coords
    assert len(test_coords) == 3
    for c in test_coords:
        assert isinstance(c, Vec3)

def test_items(turbine_map_fixture):
    """
    The class should return a list containing a dictionary of Vec3-Turbine pairs
    """
    test_items = turbine_map_fixture.items
    assert len(test_items) == 3
    for vec3, turbine in test_items:
        assert isinstance(vec3, Vec3)
        assert isinstance(turbine, Turbine)
