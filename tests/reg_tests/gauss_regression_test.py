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
VELOCITY_MODEL = "gauss_legacy"
DEFLECTION_MODEL = "gauss"

baseline = [
    (0.7634300, 1695368.6455473, 0.2568077, 7.9803783),
    (0.8436903, 651362.9121753, 0.3023199, 5.8384411),
    (0.8385498, 686209.4710003, 0.2990957, 5.9388958),
]

yawed_baseline = [
    (0.7605249, 1683956.3885389, 0.2548147, 7.9803783),
    (0.8419285, 663305.9063892, 0.3012090, 5.8728701),
    (0.8383413, 687622.7755572, 0.2989660, 5.9429700),
]

gch_baseline = [
    (0.7605249, 1683956.3885389, 0.2548147, 7.9803783),
    (0.8409478, 669953.8921404, 0.3005933, 5.8920347),
    (0.8370054, 696678.9863587, 0.2981370, 5.9690770),
]

yaw_added_recovery_baseline = [
    (0.7605249, 1683956.3885389, 0.2548147, 7.9803783),
    (0.8409481, 669952.1496101, 0.3005934, 5.8920297),
    (0.8372996, 694684.9361087, 0.2983193, 5.9633286),
]

secondary_steering_baseline = [
    (0.7605249, 1683956.3885389, 0.2548147, 7.9803783),
    (0.8419282, 663307.6815433, 0.3012088, 5.8728752),
    (0.8380415, 689655.4839532, 0.2987797, 5.9488299),
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
