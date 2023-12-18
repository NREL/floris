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

from floris.simulation import FlowField, TurbineGrid
from tests.conftest import N_FINDEX, N_TURBINES


def test_n_findex(flow_field_fixture):
    assert flow_field_fixture.n_findex == N_FINDEX


def test_initialize_velocity_field(flow_field_fixture, turbine_grid_fixture: TurbineGrid):
    flow_field_fixture.wind_shear = 1.0
    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)

    # Check the shape of the velocity arrays: u_initial, v_initial, w_initial  and u, v, w
    # Dimensions are (# findex, # turbines, N grid points, M grid points)
    assert np.shape(flow_field_fixture.u_sorted)[0] == flow_field_fixture.n_findex
    assert np.shape(flow_field_fixture.u_sorted)[1] == N_TURBINES
    assert np.shape(flow_field_fixture.u_sorted)[2] == turbine_grid_fixture.grid_resolution
    assert np.shape(flow_field_fixture.u_sorted)[3] == turbine_grid_fixture.grid_resolution

    # Check that the wind speed profile was created correctly. By setting the shear
    # exponent to 1.0 above, the shear profile is a linear function of height and
    # the points on the turbine rotor are equally spaced about the rotor.
    # This means that their average should equal the wind speed at the center
    # which is the input wind speed.
    shape = np.shape(flow_field_fixture.u_sorted[0, 0, :, :])
    n_elements = shape[0] * shape[1]
    average = (
        np.sum(flow_field_fixture.u_sorted[:, 0, :, :], axis=(-2, -1))
        / np.array([n_elements])
    )
    assert np.array_equal(average, flow_field_fixture.wind_speeds)


def test_asdict(flow_field_fixture: FlowField, turbine_grid_fixture: TurbineGrid):

    flow_field_fixture.initialize_velocity_field(turbine_grid_fixture)
    dict1 = flow_field_fixture.as_dict()

    new_ff = FlowField.from_dict(dict1)
    new_ff.initialize_velocity_field(turbine_grid_fixture)
    dict2 = new_ff.as_dict()

    assert dict1 == dict2
