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
VELOCITY_MODEL = "curl"
DEFLECTION_MODEL = "curl"

baseline = [
    (0.7636967, 1689187.1164557, 0.2569448, 7.9700630),
    (0.8326933, 731401.6677331, 0.2954843, 6.0592226),
    (0.8589734, 547760.8043000, 0.3122325, 5.5397800),
]

yawed_baseline = [
    (0.7607906, 1677789.6107559, 0.2549496, 7.9700630),
    (0.8307488, 748494.6942714, 0.2942993, 6.1014089),
    (0.8566472, 563529.9139905, 0.3106902, 5.5852387),
]

wd315_baseline = [
    (0.7636967, 1689187.1164557, 0.2569448, 7.9700630),
    (0.7636967, 1689187.1163832, 0.2569448, 7.9700630),
    (0.7636967, 1689187.1164277, 0.2569448, 7.9700630),
]


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
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])


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
    assert approx(turbine.Ct, rel=1e-5) == second_waked_baseline[0]
    assert approx(turbine.power, rel=1e-5) == second_waked_baseline[1]
    assert approx(turbine.aI, rel=1e-5) == second_waked_baseline[2]
    assert approx(turbine.average_velocity, rel=1e-5) == second_waked_baseline[3]


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
        baseline = yawed_baseline
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])


def test_regression_multiple_calc_wake(sample_inputs_fixture):
    """
    Verify turbine values stay the same with repeated (3x) calculate_wake calls.
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
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])

    floris.farm.flow_field.calculate_wake()
    test_results = turbines_to_array(floris.farm.turbine_map.turbines)
    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])

    floris.farm.flow_field.calculate_wake()
    test_results = turbines_to_array(floris.farm.turbine_map.turbines)
    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)
    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])


def test_change_wind_direction(sample_inputs_fixture):
    """
    Tandem turbines aligned to the wind direction, first calculated at 270 degrees wind
    direction, and then calculated at 315 degrees wind direction.
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

    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])

    floris.farm.wind_map.input_direction = [315.0]
    floris.farm.wind_map.calculate_wind_direction()
    floris.farm.turbine_map.reinitialize_turbines()
    floris.farm.flow_field.reinitialize_flow_field()

    floris.farm.flow_field.calculate_wake()

    test_results = turbines_to_array(floris.farm.turbine_map.turbines)

    if DEBUG:
        print_test_values(floris.farm.turbine_map.turbines)

    for i, turbine in enumerate(floris.farm.turbine_map.turbines):
        assert test_results[i][0] == approx(wd315_baseline[i][0])
        assert test_results[i][1] == approx(wd315_baseline[i][1])
        assert test_results[i][2] == approx(wd315_baseline[i][2])
        assert test_results[i][3] == approx(wd315_baseline[i][3])
