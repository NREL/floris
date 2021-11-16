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
VELOCITY_MODEL = "multizone"
DEFLECTION_MODEL = "jimenez"

baseline = [
    (0.7634300, 1695368.6455473, 0.2568077, 7.9803783),
    (0.8686178, 498160.7231994, 0.3187666, 5.3776582),
    (0.9336986, 256132.0247252, 0.3712547, 4.4402120),
]

yawed_baseline = [
    (0.7605249, 1683956.3885389, 0.2548147, 7.9803783),
    (0.8571693, 559990.6988904, 0.3110353, 5.5750360),
    (0.9226030, 286513.8101280, 0.3608984, 4.5782528),
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
        baseline = yawed_baseline
        assert test_results[i][0] == approx(baseline[i][0])
        assert test_results[i][1] == approx(baseline[i][1])
        assert test_results[i][2] == approx(baseline[i][2])
        assert test_results[i][3] == approx(baseline[i][3])
