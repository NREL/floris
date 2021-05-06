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

import copy

from pytest import approx

from tests.conftest import assert_results, print_test_values, turbines_to_array
from src import Floris


DEBUG = False
VELOCITY_MODEL = "multizone"
DEFLECTION_MODEL = "jimenez"

baseline = [
    (0.7655527, 1816305.2933644, 0.2579012, 7.9803783),
    (0.8681066, 534325.7405792, 0.3184143, 5.3665830),
    (0.9477858, 247452.0393393, 0.3857479, 4.2843826),
]

yawed_baseline = [
    (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
    (0.8646618, 549911.3056840, 0.3160583, 5.4198115),
    (0.9307933, 279868.4185378, 0.3684641, 4.4569688),
]

# Note: compare the yawed vs non-yawed results. The upstream turbine
# power should be lower in the yawed case. The following turbine
# powers should higher in the yawed case.


def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.floris["wake"]["properties"][
        "velocity_model"
    ] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"][
        "deflection_model"
    ] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, baseline)

def test_regression_rotation(sample_inputs_fixture):
    """
    Turbines in tandem and rotated.
    The result from 270 degrees should match the results from 360 degrees.
    """
    sample_inputs_fixture.floris["wake"]["properties"][
        "velocity_model"
    ] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"][
        "deflection_model"
    ] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    fresh_turbine = copy.deepcopy(floris.farm.turbine_map.turbines[0])
    wind_map = floris.farm.wind_map

    # Generate the unrotated baselines
    floris.farm.flow_field.calculate_wake()
    [unwaked_baseline, first_waked_baseline, second_waked_baseline] = turbines_to_array(floris.farm.turbine_map.turbines)

    # Rotate and calculate
    wind_map.input_direction = [360]
    wind_map.calculate_wind_direction()
    new_map = TurbineMap(
        [0.0, 0.0, 0.0],
        [
            10 * sample_inputs_fixture.floris["turbine"]["rotor_diameter"],
            5 * sample_inputs_fixture.floris["turbine"]["rotor_diameter"],
            0.0,
        ],
        [
            copy.deepcopy(fresh_turbine),
            copy.deepcopy(fresh_turbine),
            copy.deepcopy(fresh_turbine),
        ],
    )
    floris.farm.flow_field.reinitialize_flow_field(
        turbine_map=new_map, wind_map=wind_map
    )
    floris.farm.flow_field.calculate_wake()

    # Compare results to baaselines
    test_results = turbines_to_array(floris.farm.turbine_map.turbines)
    assert test_results[0] == approx(unwaked_baseline)
    assert test_results[1] == approx(first_waked_baseline)
    assert test_results[2] == approx(second_waked_baseline)

def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    sample_inputs_fixture.floris["wake"]["properties"][
        "velocity_model"
    ] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"][
        "deflection_model"
    ] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)

    # yaw the upstream turbine 5 degrees
    floris.farm.turbines[0].yaw_angle = 5.0
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, yawed_baseline)
