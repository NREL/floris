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
import numpy as np
from floris.simulation import FlowField, Turbine, TurbineMap, Wake, WindMap
import copy


@pytest.fixture
def flow_field_fixture(sample_inputs_fixture):
    wake = Wake(sample_inputs_fixture.wake)
    turbine = Turbine(sample_inputs_fixture.turbine)
    turbine_map = TurbineMap(
        [0.0, 100.0],
        [0.0, 0.0],
        [copy.deepcopy(turbine), copy.deepcopy(turbine)]
    )
    farm_prop = sample_inputs_fixture.farm["properties"]
    wind_map = WindMap(
        wind_speed = farm_prop["wind_speed"],
        layout_array = (farm_prop["layout_x"], farm_prop["layout_y"]),
        wind_layout = (farm_prop["wind_x"], farm_prop["wind_y"]),
        turbulence_intensity = farm_prop["turbulence_intensity"],
        wind_direction = farm_prop["wind_direction"]
    )
    return FlowField(
        farm_prop["wind_shear"],
        farm_prop["wind_veer"],
        farm_prop["air_density"],
        wake,
        turbine_map,
        wind_map,
        farm_prop["specified_wind_height"]
    )

def test_instantiation(flow_field_fixture):
    """
    The class should initialize with the standard inputs
    """
    assert type(flow_field_fixture) is FlowField

def test_discretize_domain(flow_field_fixture):
    """
    The class should discretize the domain on initialization with three
    component-arrays each of type np.ndarray and size (100, 100, 50)
    """
    x, y, z = flow_field_fixture._discretize_turbine_domain()
    assert np.shape(x) == (2, 5, 5) and type(x) is np.ndarray \
        and np.shape(y) == (2, 5, 5) and type(y) is np.ndarray \
        and np.shape(z) == (2, 5, 5) and type(z) is np.ndarray

def test_2x_calculate_wake(flow_field_fixture):
    """
    Calling `calculate_wake` multiple times should result in the same
    calculation.

    Args:
        flow_field_fixture (pytest.fixture): The pytest fixture for the class.
    """
    n_turbines = len(flow_field_fixture.turbine_map.turbines)

    # Establish a container for the data in successive calls to calculate wake
    calculate_wake_results = {
        1: [None]*n_turbines,
        2: [None]*n_turbines
    }

    # Do the first call
    flow_field_fixture.calculate_wake()
    for i, turbine in enumerate(flow_field_fixture.turbine_map.turbines):
        calculate_wake_results[1][i] = {
            "Cp": turbine.Cp,
            "Ct": turbine.Ct,
            "power": turbine.power,
            "aI": turbine.aI,
            "average_velocity": turbine.average_velocity
        }

    # Do the second call
    flow_field_fixture.calculate_wake()
    for i, turbine in enumerate(flow_field_fixture.turbine_map.turbines):
        calculate_wake_results[2][i] = {
            "Cp": turbine.Cp,
            "Ct": turbine.Ct,
            "power": turbine.power,
            "aI": turbine.aI,
            "average_velocity": turbine.average_velocity
        }

    # Compare the results
    assert calculate_wake_results[1] == calculate_wake_results[2]
