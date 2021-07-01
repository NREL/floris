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

from tests.conftest import print_test_values, turbines_to_array
from floris.simulation import Floris, TurbineMap


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = [
    (0.7655527, 1816305.2933644, 0.2579012, 7.9803783),
    (0.8364885, 745683.5817378, 0.2978172, 5.9488213),
    (0.8361129, 749047.5041809, 0.2975851, 5.9569084),
]

yawed_baseline = [
    (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
    (0.8347551, 761206.9452575, 0.2967484, 5.9861402),
    (0.8358036, 751817.5642544, 0.2973942, 5.9635677),
]

gch_baseline = [
    (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
    (0.8343969, 764414.9030250, 0.2965282, 5.9938523),
    (0.8351941, 757275.4122748, 0.2970185, 5.9766886),
]

yaw_added_recovery_baseline = [
    (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
    (0.8343970, 764413.4003995, 0.2965283, 5.9938487),
    (0.8353932, 755492.1572634, 0.2971412, 5.9724016),
]

secondary_steering_baseline = [
    (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
    (0.8347549, 761208.4605523, 0.2967483, 5.9861439),
    (0.8356027, 753616.0833354, 0.2972703, 5.9678914),
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

    for i in range(len(floris.farm.turbine_map.turbines)):
        check = baseline
        assert test_results[i][0] == approx(check[i][0])
        assert test_results[i][1] == approx(check[i][1])
        assert test_results[i][2] == approx(check[i][2])
        assert test_results[i][3] == approx(check[i][3])


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

    # unrotated
    floris.farm.flow_field.calculate_wake()
    turbine = floris.farm.turbine_map.turbines[0]
    unwaked_baseline = (
        turbine.Ct,
        turbine.power,
        turbine.aI,
        turbine.average_velocity,
    )
    turbine = floris.farm.turbine_map.turbines[1]
    first_waked_baseline = (
        turbine.Ct,
        turbine.power,
        turbine.aI,
        turbine.average_velocity,
    )
    turbine = floris.farm.turbine_map.turbines[2]
    second_waked_baseline = (
        turbine.Ct,
        turbine.power,
        turbine.aI,
        turbine.average_velocity,
    )

    # rotated
    wind_map.input_direction = [360]
    wind_map.calculate_wind_direction()
    new_map = TurbineMap(
        [0.0, 0.0, 0.0],
        [
            10
            * sample_inputs_fixture.floris["turbine"]["properties"]["rotor_diameter"],
            5 * sample_inputs_fixture.floris["turbine"]["properties"]["rotor_diameter"],
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

    turbine = floris.farm.turbine_map.turbines[0]
    assert approx(turbine.Ct) == unwaked_baseline[0]
    assert approx(turbine.power) == unwaked_baseline[1]
    assert approx(turbine.aI) == unwaked_baseline[2]
    assert approx(turbine.average_velocity) == unwaked_baseline[3]

    turbine = floris.farm.turbine_map.turbines[1]
    assert approx(turbine.Ct) == first_waked_baseline[0]
    assert approx(turbine.power) == first_waked_baseline[1]
    assert approx(turbine.aI) == first_waked_baseline[2]
    assert approx(turbine.average_velocity) == first_waked_baseline[3]

    turbine = floris.farm.turbine_map.turbines[2]
    assert approx(turbine.Ct) == second_waked_baseline[0]
    assert approx(turbine.power) == second_waked_baseline[1]
    assert approx(turbine.aI) == second_waked_baseline[2]
    assert approx(turbine.average_velocity) == second_waked_baseline[3]


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

    for i in range(len(floris.farm.turbine_map.turbines)):
        check = yawed_baseline
        assert test_results[i][0] == approx(check[i][0])
        assert test_results[i][1] == approx(check[i][1])
        assert test_results[i][2] == approx(check[i][2])
        assert test_results[i][3] == approx(check[i][3])


def test_regression_gch(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    sample_inputs_fixture.floris["wake"]["properties"][
        "velocity_model"
    ] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"][
        "deflection_model"
    ] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.turbines[0].yaw_angle = 5.0

    # With GCH off (via conftest), GCH should be same as Gauss
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    for i in range(len(floris.farm.turbine_map.turbines)):
        check = yawed_baseline
        assert test_results[i][0] == approx(check[i][0])
        assert test_results[i][1] == approx(check[i][1])
        assert test_results[i][2] == approx(check[i][2])
        assert test_results[i][3] == approx(check[i][3])

    # With GCH on, the results should change
    floris.farm.wake.deflection_model.use_secondary_steering = True
    floris.farm.wake.velocity_model.use_yaw_added_recovery = True
    floris.farm.wake.velocity_model.calculate_VW_velocities = True
    floris.farm.flow_field.reinitialize_flow_field()
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    for i in range(len(floris.farm.turbine_map.turbines)):
        check = gch_baseline
        assert test_results[i][0] == approx(check[i][0])
        assert test_results[i][1] == approx(check[i][1])
        assert test_results[i][2] == approx(check[i][2])
        assert test_results[i][3] == approx(check[i][3])


def test_regression_yaw_added_recovery(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    sample_inputs_fixture.floris["wake"]["properties"][
        "velocity_model"
    ] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"][
        "deflection_model"
    ] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.turbines[0].yaw_angle = 5.0

    # Enable yaw-added recorvery
    floris.farm.wake.velocity_model.use_yaw_added_recovery = True
    floris.farm.wake.velocity_model.calculate_VW_velocities = True
    floris.farm.flow_field.reinitialize_flow_field()
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    for i in range(len(floris.farm.turbine_map.turbines)):
        check = yaw_added_recovery_baseline
        assert test_results[i][0] == approx(check[i][0])
        assert test_results[i][1] == approx(check[i][1])
        assert test_results[i][2] == approx(check[i][2])
        assert test_results[i][3] == approx(check[i][3])


def test_regression_secondary_steering(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """
    sample_inputs_fixture.floris["wake"]["properties"][
        "velocity_model"
    ] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"][
        "deflection_model"
    ] = DEFLECTION_MODEL
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    floris.farm.turbines[0].yaw_angle = 5.0

    # Enable secondary steering
    floris.farm.wake.deflection_model.use_secondary_steering = True
    floris.farm.wake.velocity_model.calculate_VW_velocities = True
    floris.farm.flow_field.reinitialize_flow_field()
    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    for i in range(len(floris.farm.turbine_map.turbines)):
        check = secondary_steering_baseline
        assert test_results[i][0] == approx(check[i][0])
        assert test_results[i][1] == approx(check[i][1])
        assert test_results[i][2] == approx(check[i][2])
        assert test_results[i][3] == approx(check[i][3])
