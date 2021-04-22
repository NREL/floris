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
from src import Floris, TurbineMap


DEBUG = False
VELOCITY_MODEL = "curl"
DEFLECTION_MODEL = "curl"

baseline = [
    (0.7655987, 1808879.0573334, 0.2579249, 7.9700630),
    (0.8315292, 790095.3264943, 0.2947740, 6.0555893),
    (0.8568269, 585359.6178647, 0.3108089, 5.5408759),
]

yawed_baseline = [
    (0.7626853, 1795186.5605035, 0.2559142, 7.9700630),
    (0.8295616, 807715.7153082, 0.2935790, 6.0979495),
    (0.8538732, 598723.1123378, 0.3088673, 5.5865154),
]

wd315_baseline = [
    (0.7655987, 1808879.0573334, 0.2579249, 7.9700630),
    (0.7655987, 1808879.0573334, 0.2579249, 7.9700630),
    (0.7655987, 1808879.0573334, 0.2579249, 7.9700630),
]

# Note: compare the yawed vs non-yawed results. The upstream turbine
# power should be lower in the yawed case. The following turbine
# powers should higher in the yawed case.

def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
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
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
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
    assert test_results[0] == approx(unwaked_baseline, rel=1e-5)
    assert test_results[1] == approx(first_waked_baseline, rel=1e-5)
    assert test_results[2] == approx(second_waked_baseline, rel=1e-5)

def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)

    # yaw the upstream turbine 5 degrees
    floris.farm.turbines[0].yaw_angle = 5.0
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, yawed_baseline)

def test_regression_multiple_calc_wake(sample_inputs_fixture):
    """
    Verify turbine values stay the same with repeated (3x) calculate_wake calls.
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)

    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, baseline)

    floris.farm.flow_field.calculate_wake()
    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, baseline)

    floris.farm.flow_field.calculate_wake()
    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, baseline)

def test_change_wind_direction(sample_inputs_fixture):
    """
    Tandem turbines aligned to the wind direction, first calculated at 270 degrees wind
    direction, and then calculated at 315 degrees wind direction.
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, baseline)

    floris.farm.wind_map.input_direction = [315.0]
    floris.farm.wind_map.calculate_wind_direction()
    floris.farm.turbine_map.reinitialize_turbines()
    floris.farm.flow_field.reinitialize_flow_field()

    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    assert_results(test_results, wd315_baseline)
