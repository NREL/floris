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


import pytest
import numpy as np
from src import FlowFieldGrid, TurbineGrid
from src.utilities import Vec3


GRID_RESOLUTION = 2
N_TURBINES = 3
X_COORDS = [
    0.0,
    5 * 126.0,
    10 * 126.0
]
Y_COORDS = [
    0.0,
    0.0,
    0.0
]
Z_COORDS = [
    90.0,
    90.0,
    90.0
]


@pytest.fixture
def turbine_grid_fixture(sample_inputs_fixture) -> TurbineGrid:
    turbine_coordinates = list(zip(X_COORDS, Y_COORDS, Z_COORDS))
    turbine_coordinates = [Vec3(c) for c in turbine_coordinates]
    return TurbineGrid(
        turbine_coordinates,
        sample_inputs_fixture.turbine["rotor_diameter"],
        sample_inputs_fixture.farm["reference_wind_height"],
        GRID_RESOLUTION
    )


@pytest.fixture
def flow_field_grid_fixture(sample_inputs_fixture):
    n_turbines = len(sample_inputs_fixture.farm["layout_x"])
    turbine_coordinates = []
    for i in range(n_turbines):
        turbine_coordinates.append(
            Vec3([
                sample_inputs_fixture.farm["layout_x"][i],
                sample_inputs_fixture.farm["layout_y"][i],
                0.0
            ])
        )
    return FlowFieldGrid(
        turbine_coordinates,
        sample_inputs_fixture.turbine["rotor_diameter"],
        sample_inputs_fixture.farm["reference_wind_height"],
        Vec3([2,2,2])
    )


def test_turbine_set_grid(turbine_grid_fixture):
    expected_x_grid = [[[0.0, 0.0], [0.0, 0.0]], [[630.0, 630.0], [630.0, 630.0]], [[1260.0, 1260.0], [1260.0, 1260.0]]]
    expected_y_grid = [[[-31.5, 31.5], [-31.5, 31.5]], [[-31.5, 31.5], [-31.5, 31.5]], [[-31.5, 31.5], [-31.5, 31.5]]]
    expected_z_grid = [[[58.5, 121.5], [58.5, 121.5]], [[58.5, 121.5], [58.5, 121.5]], [[58.5, 121.5], [58.5, 121.5]]] 

    # subtract the test and expected values which should result in 0's
    # then, search for any elements that are true and negate the results
    # if an element is zero, the not will return true
    # if an element is non-zero, the not will return false
    assert not np.any( turbine_grid_fixture.x - expected_x_grid )
    assert not np.any( turbine_grid_fixture.y - expected_y_grid )
    assert not np.any( turbine_grid_fixture.z - expected_z_grid )


def test_turbinegrid_dimensions(turbine_grid_fixture):
    assert np.shape(turbine_grid_fixture.x) == (N_TURBINES, GRID_RESOLUTION, GRID_RESOLUTION)
    assert np.shape(turbine_grid_fixture.y) == (N_TURBINES, GRID_RESOLUTION, GRID_RESOLUTION)
    assert np.shape(turbine_grid_fixture.z) == (N_TURBINES, GRID_RESOLUTION, GRID_RESOLUTION)


def test_flow_field_set_bounds(flow_field_grid_fixture):
    assert flow_field_grid_fixture.xmin == -252.0
    assert flow_field_grid_fixture.xmax == 2520.0
    assert flow_field_grid_fixture.ymin == -252.0
    assert flow_field_grid_fixture.ymax == 252.0
    assert flow_field_grid_fixture.zmin == 0.1
    assert flow_field_grid_fixture.zmax == 540


def test_flow_field_set_grid(flow_field_grid_fixture):
    assert [flow_field_grid_fixture.x[0][0][0], flow_field_grid_fixture.y[0][0][0], flow_field_grid_fixture.z[0][0][0]] == [ -252.0, -252.0, 0.1]
    assert [flow_field_grid_fixture.x[1][0][0], flow_field_grid_fixture.y[0][0][0], flow_field_grid_fixture.z[0][0][0]] == [ 2520.0, -252.0, 0.1]
    assert [flow_field_grid_fixture.x[0][0][0], flow_field_grid_fixture.y[0][1][0], flow_field_grid_fixture.z[0][0][0]] == [ -252.0,  252.0, 0.1]
    assert [flow_field_grid_fixture.x[1][0][0], flow_field_grid_fixture.y[0][1][0], flow_field_grid_fixture.z[0][0][0]] == [ 2520.0,  252.0, 0.1]
    assert [flow_field_grid_fixture.x[0][0][0], flow_field_grid_fixture.y[0][0][0], flow_field_grid_fixture.z[0][0][1]] == [ -252.0, -252.0, 540.0]
    assert [flow_field_grid_fixture.x[1][0][0], flow_field_grid_fixture.y[0][0][0], flow_field_grid_fixture.z[0][0][1]] == [ 2520.0, -252.0, 540.0]
    assert [flow_field_grid_fixture.x[0][0][0], flow_field_grid_fixture.y[0][1][0], flow_field_grid_fixture.z[0][0][1]] == [ -252.0,  252.0, 540.0]
    assert [flow_field_grid_fixture.x[1][0][0], flow_field_grid_fixture.y[0][1][0], flow_field_grid_fixture.z[0][0][1]] == [ 2520.0,  252.0, 540.0]
