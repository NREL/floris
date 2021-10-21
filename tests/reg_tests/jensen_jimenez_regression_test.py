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
    (7.9803783, 0.7634300, 1695368.6455473, 0.2568077),
    (6.1586693, 0.8281095, 771695.5183645, 0.2927016),
    (5.6649575, 0.8525678, 591183.4224051, 0.3080155),
]

yawed_baseline = [
    (7.9803783, 0.7605249, 1683956.3885389, 0.2548147),
    (6.1728072, 0.8274579, 777423.9137261, 0.2923090),
    (5.6709666, 0.8522603, 593267.9301046, 0.3078154),
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
    floris.go()

    turbines = floris.farm.turbines
    n_turbines = len(turbines)

    test_results = []

    velocities = floris.flow_field.u[:, :, :, :]
    yaw_angles = n_turbines * [0.0]
    thrust_interpolation_func = [t.fCt for t in turbines]
    power_interpolation_func = [t.power_interp for t in turbines]
    power_exponent = [t.pP for t in turbines]

    farm_avg_velocities = average_velocity(
        velocities[0, :, :, :],
        ix_filter=list(range(n_turbines))
    )
    farm_cts = Ct(
        velocities[0, :, :, :],
        yaw_angles,
        thrust_interpolation_func,
        ix_filter=list(range(n_turbines))
    )
    farm_powers = power(
        n_turbines * [floris.flow_field.air_density],
        velocities[0, :, :, :],
        yaw_angles,
        power_exponent,
        power_interpolation_func,
        ix_filter=list(range(n_turbines))
    )
    farm_axial_inductions = axial_induction(
        velocities[0, :, :, :],
        yaw_angles,
        thrust_interpolation_func,
        ix_filter=list(range(n_turbines))
    )
    for i in range(n_turbines):
        this_turbine = [
            farm_avg_velocities[i],
            farm_cts[i],
            farm_powers[i],
            farm_axial_inductions[i]
        ]
        test_results.append(this_turbine)

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
        )

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
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris(input_dict=sample_inputs_fixture.floris)

    # yaw the upstream turbine 5 degrees
    floris.farm.set_yaw_angles([5.0, 0.0, 0.0], floris.flow_field.n_wind_speeds, 1) # TODO: n_wind_directions

    floris.go()

    turbines = floris.farm.turbines
    n_turbines = len(turbines)


    test_results = []

    velocities = floris.flow_field.u[:, :, :, :]
    yaw_angles = floris.farm.farm_controller.yaw_angles
    thrust_interpolation_func = [t.fCt for t in turbines]
    power_interpolation_func = [t.power_interp for t in turbines]
    power_exponent = [t.pP for t in turbines]

    farm_avg_velocities = average_velocity(
        velocities[0, :, :, :],
        ix_filter=list(range(n_turbines))
    )
    farm_cts = Ct(
        velocities[0, :, :, :],
        yaw_angles,
        thrust_interpolation_func,
        ix_filter=list(range(n_turbines))
    )
    farm_powers = power(
        n_turbines * [floris.flow_field.air_density],
        velocities[0, :, :, :],
        yaw_angles,
        power_exponent,
        power_interpolation_func,
        ix_filter=list(range(n_turbines))
    )
    farm_axial_inductions = axial_induction(
        velocities[0, :, :, :],
        yaw_angles,
        thrust_interpolation_func,
        ix_filter=list(range(n_turbines))
    )
    for i in range(n_turbines):
        this_turbine = [
            farm_avg_velocities[i],
            farm_cts[i],
            farm_powers[i],
            farm_axial_inductions[i]
        ]
        test_results.append(this_turbine)

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
        )

    assert_results(test_results, yawed_baseline)
