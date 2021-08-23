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

import copy
import numpy as np
from pytest import approx

from tests.conftest import assert_results, print_test_values, turbines_to_array
from src import Floris
from src.turbine import power, Ct, axial_induction, average_velocity

import time

DEBUG = False
VELOCITY_MODEL = "jensen"
DEFLECTION_MODEL = "jimenez"

baseline = [
    (0.7634300, 1695368.6455473, 0.2568077, 7.9803783),
    (0.8281095, 771695.5183645, 0.2927016, 6.1586693),
    (0.8525678, 591183.4224051, 0.3080155, 5.6649575),
]

yawed_baseline = [
    (0.7605249, 1683956.3885389, 0.2548147, 7.9803783),
    (0.8274579, 777423.9137261, 0.2923090, 6.1728072),
    (0.8522603, 593267.9301046, 0.3078154, 5.6709666),
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

    tic = time.perf_counter()
    floris = Floris(input_dict=sample_inputs_fixture.floris)
    toc = time.perf_counter()
    print(f"Initialization in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    floris.go()
    toc = time.perf_counter()
    print(f"Calculation in {toc - tic:0.4f} seconds")
    
    n_turbines = 3
    turbine1 = floris.farm.turbines[0]
    test_results = []
    for i in range(n_turbines):
        # print(floris.flow_field.u)
        ave_vel = average_velocity(floris.flow_field.u[i, 0, :, :])
        thrust = Ct(ave_vel, 0.0, turbine1.fCt)
        pwr = power(floris.flow_field.air_density, ave_vel, 0.0, turbine1.pP, turbine1.power_interp)
        ai = axial_induction(thrust, 0.0)
        this_turbine = [
                thrust,
                pwr,
                ai,
                ave_vel
        ]
        test_results.append(this_turbine)

    # print(floris.flow_field.u_initial)
    # if DEBUG:
    #     print_test_values(floris.farm.turbine_map.turbines)

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
