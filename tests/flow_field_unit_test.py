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
from src import FlowField
from .grid_unit_test import turbine_grid_fixture, N_TURBINES

@pytest.fixture
def flow_field_fixture(sample_inputs_fixture):
    farm_dict = sample_inputs_fixture.farm
    return FlowField(farm_dict)


def test_n_wind_speeds(flow_field_fixture):
    assert flow_field_fixture.n_wind_speeds > 0


def test_initialize_velocity_field(flow_field_fixture, turbine_grid_fixture):
    flow_field_fixture.wind_shear = 1.0
    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)

    # Check the shape of the velocity arrays: u_initial, v_initial, w_initial  and u, v, w
    # Dimensions are (# wind speeds, # turbines, N grid points, M grid points)
    assert np.shape(flow_field_fixture.u)[0] == flow_field_fixture.n_wind_speeds
    assert np.shape(flow_field_fixture.u)[1] == N_TURBINES
    assert np.shape(flow_field_fixture.u)[2] == turbine_grid_fixture.grid_resolution
    assert np.shape(flow_field_fixture.u)[3] == turbine_grid_fixture.grid_resolution

    # Check that the wind speed profile was created correctly. By setting the shear
    # exponent to 1.0 above, the shear profile is a linear function of height and
    # the points on the rurbine rotor are equally spaced about the rotor.
    # This means that their average should equal the wind speed at the center
    # which is the input wind speed.
    shape = np.shape(flow_field_fixture.u)
    n_elements = shape[1] * shape[2] * shape[3]
    average = np.sum(flow_field_fixture.u, axis=(1,2,3)) / np.array([n_elements, n_elements])
    assert np.array_equal(average, flow_field_fixture.wind_speeds)

    # assert False