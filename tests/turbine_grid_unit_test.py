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

from floris.simulation import TurbineGrid
from tests.conftest import (
    N_FINDEX,
    N_TURBINES,
    TURBINE_GRID_RESOLUTION,
)


# def test_from_dict_as_dict(turbine_grid_fixture):
#     grid_dict = turbine_grid_fixture.as_dict()
#     new_grid = TurbineGrid.from_dict(grid_dict)
#     assert new_grid == turbine_grid_fixture


def test_set_grid(turbine_grid_fixture):
    expected_x_grid = [
        [[0.0, 0.0], [0.0, 0.0]],
        [[630.0, 630.0], [630.0, 630.0]],
        [[1260.0, 1260.0], [1260.0, 1260.0]]
    ]
    expected_y_grid = [
        [[-31.5, -31.5], [31.5, 31.5]],
        [[-31.5, -31.5], [31.5, 31.5]],
        [[-31.5, -31.5], [31.5, 31.5]]
    ]
    expected_z_grid = [
        [[58.5, 121.5], [58.5, 121.5]],
        [[58.5, 121.5], [58.5, 121.5]],
        [[58.5, 121.5], [58.5, 121.5]]
    ]

    # subtract the test and expected values which should result in 0's
    # then, search for any elements that are true and negate the results
    # if an element is zero, the not will return true
    # if an element is non-zero, the not will return false
    np.testing.assert_array_equal(turbine_grid_fixture.x_sorted[0], expected_x_grid)
    np.testing.assert_array_equal(turbine_grid_fixture.y_sorted[0], expected_y_grid)
    np.testing.assert_array_equal(turbine_grid_fixture.z_sorted[0], expected_z_grid)

    # These should have the following shape:
    # (n findex, n turbines, grid resolution, grid resolution)
    expected_shape = (N_FINDEX,N_TURBINES,TURBINE_GRID_RESOLUTION,TURBINE_GRID_RESOLUTION)
    assert np.shape(turbine_grid_fixture.x_sorted) == expected_shape
    assert np.shape(turbine_grid_fixture.y_sorted) == expected_shape
    assert np.shape(turbine_grid_fixture.z_sorted) == expected_shape
    assert np.shape(turbine_grid_fixture.x_sorted_inertial_frame) == expected_shape
    assert np.shape(turbine_grid_fixture.y_sorted_inertial_frame) == expected_shape
    assert np.shape(turbine_grid_fixture.z_sorted_inertial_frame) == expected_shape


def test_dimensions(turbine_grid_fixture):
    assert np.shape(turbine_grid_fixture.x_sorted) == (
        N_FINDEX,
        N_TURBINES,
        TURBINE_GRID_RESOLUTION,
        TURBINE_GRID_RESOLUTION
    )
    assert np.shape(turbine_grid_fixture.y_sorted) == (
        N_FINDEX,
        N_TURBINES,
        TURBINE_GRID_RESOLUTION,
        TURBINE_GRID_RESOLUTION
    )
    assert np.shape(turbine_grid_fixture.z_sorted) == (
        N_FINDEX,
        N_TURBINES,
        TURBINE_GRID_RESOLUTION,
        TURBINE_GRID_RESOLUTION
    )


def test_dynamic_properties(turbine_grid_fixture):
    assert turbine_grid_fixture.n_turbines == N_TURBINES
    assert turbine_grid_fixture.n_findex == N_FINDEX

    turbine_grid_fixture.turbine_coordinates = np.append(
        turbine_grid_fixture.turbine_coordinates,
        np.array([[100.0, 200.0, 300.0]]),
        axis=0
    )
    assert turbine_grid_fixture.n_turbines == N_TURBINES + 1

    turbine_grid_fixture.wind_directions = [*turbine_grid_fixture.wind_directions, 0.0]
    assert turbine_grid_fixture.n_findex == N_FINDEX + 1
