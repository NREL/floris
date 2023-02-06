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
from tests.conftest import (
    N_TURBINES,
    N_WIND_DIRECTIONS,
    N_WIND_SPEEDS,
    TURBINE_GRID_RESOLUTION,
)


# TODO: test the dimension expansion


def test_turbinegrid_set_grid(turbine_grid_fixture):
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
    assert not np.any(turbine_grid_fixture.x_sorted[0, 0] - expected_x_grid)
    assert not np.any(turbine_grid_fixture.y_sorted[0, 0] - expected_y_grid)
    assert not np.any(turbine_grid_fixture.z_sorted[0, 0] - expected_z_grid)


def test_turbinegrid_dimensions(turbine_grid_fixture):
    assert np.shape(turbine_grid_fixture.x_sorted) == (
        N_WIND_DIRECTIONS,
        N_WIND_SPEEDS,
        N_TURBINES,
        TURBINE_GRID_RESOLUTION,
        TURBINE_GRID_RESOLUTION
    )
    assert np.shape(turbine_grid_fixture.y_sorted) == (
        N_WIND_DIRECTIONS,
        N_WIND_SPEEDS,
        N_TURBINES,
        TURBINE_GRID_RESOLUTION,
        TURBINE_GRID_RESOLUTION
    )
    assert np.shape(turbine_grid_fixture.z_sorted) == (
        N_WIND_DIRECTIONS,
        N_WIND_SPEEDS,
        N_TURBINES,
        TURBINE_GRID_RESOLUTION,
        TURBINE_GRID_RESOLUTION
    )


def test_turbinegrid_dynamic_properties(turbine_grid_fixture):
    assert turbine_grid_fixture.n_turbines == N_TURBINES
    assert turbine_grid_fixture.n_wind_speeds == N_WIND_SPEEDS
    assert turbine_grid_fixture.n_wind_directions == N_WIND_DIRECTIONS

    # TODO: @Rob @Chris This breaks n_turbines since the validator is not run.
    # Is this case ok? Do we enforce that turbine_coordinates must be set by =?
    # turbine_grid_fixture.turbine_coordinates.append(Vec3([100.0, 200.0, 300.0]))
    # assert turbine_grid_fixture.n_turbines == N_TURBINES + 1

    turbine_grid_fixture.turbine_coordinates = [
        *turbine_grid_fixture.turbine_coordinates, Vec3([100.0, 200.0, 300.0])
    ]
    assert turbine_grid_fixture.n_turbines == N_TURBINES + 1

    turbine_grid_fixture.wind_speeds = [*turbine_grid_fixture.wind_speeds, 0.0]
    assert turbine_grid_fixture.n_wind_speeds == N_WIND_SPEEDS + 1

    turbine_grid_fixture.wind_directions = [*turbine_grid_fixture.wind_directions, 0.0]
    assert turbine_grid_fixture.n_wind_directions == N_WIND_DIRECTIONS + 1



# def test_flow_field_set_bounds(flow_field_grid_fixture):
#     assert flow_field_grid_fixture.xmin == -252.0
#     assert flow_field_grid_fixture.xmax == 2520.0
#     assert flow_field_grid_fixture.ymin == -252.0
#     assert flow_field_grid_fixture.ymax == 252.0
#     assert flow_field_grid_fixture.zmin == 0.1
#     assert flow_field_grid_fixture.zmax == 540


# def test_flow_field_set_grid(flow_field_grid_fixture):
#     assert (
#         [
#             flow_field_grid_fixture.x[0][0][0],
#             flow_field_grid_fixture.y[0][0][0],
#             flow_field_grid_fixture.z[0][0][0]
#         ]
#         == [ -252.0, -252.0, 0.1]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[1][0][0],
#             flow_field_grid_fixture.y[0][0][0],
#             flow_field_grid_fixture.z[0][0][0]
#         ]
#         == [ 2520.0, -252.0, 0.1]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[0][0][0],
#             flow_field_grid_fixture.y[0][1][0],
#             flow_field_grid_fixture.z[0][0][0]
#         ]
#         == [ -252.0,  252.0, 0.1]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[1][0][0],
#             flow_field_grid_fixture.y[0][1][0],
#             flow_field_grid_fixture.z[0][0][0]
#         ]
#         == [ 2520.0,  252.0, 0.1]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[0][0][0],
#             flow_field_grid_fixture.y[0][0][0],
#             flow_field_grid_fixture.z[0][0][1]
#         ]
#         == [ -252.0, -252.0, 540.0]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[1][0][0],
#             flow_field_grid_fixture.y[0][0][0],
#             flow_field_grid_fixture.z[0][0][1]
#         ]
#         == [ 2520.0, -252.0, 540.0]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[0][0][0],
#             flow_field_grid_fixture.y[0][1][0],
#             flow_field_grid_fixture.z[0][0][1]
#         ]
#         == [ -252.0,  252.0, 540.0]
#     )
#     assert (
#         [
#             flow_field_grid_fixture.x[1][0][0],
#             flow_field_grid_fixture.y[0][1][0],
#             flow_field_grid_fixture.z[0][0][1]
#         ]
#         == [ 2520.0,  252.0, 540.0]
#     )
