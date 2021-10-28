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

import numpy as np

from src import Floris
from src.turbine import Ct, power, axial_induction, average_velocity
from tests.conftest import print_test_values, turbines_to_array, assert_results_arrays


DEBUG = False
VELOCITY_MODEL = "jensen"
DEFLECTION_MODEL = "jimenez"


baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7634300, 1695368.6455473, 0.2568077],
            [6.1586693, 0.8281095, 771695.5183645, 0.2927016],
            [5.6649575, 0.8525678, 591183.4224051, 0.3080155],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7625731, 2413659.0651694, 0.2563676],
            [6.9320149, 0.7949935, 1111075.5222317, 0.2736118],
            [6.5096913, 0.8119868, 914506.7978006, 0.2831975],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7527803, 3306006.9741814, 0.2513940],
            [7.7463403, 0.7694798, 1555119.6348506, 0.2599374],
            [7.3515939, 0.7807184, 1328908.6335441, 0.2658625],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.3885389, 0.2548147],
            [6.1728072, 0.8274579, 777423.9137261, 0.2923090],
            [5.6709666, 0.8522603, 593267.9301046, 0.3078154],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397237.3791443, 0.2543815],
            [6.9478646, 0.7943557, 1118452.7210795, 0.2732599],
            [6.5163235, 0.8117199, 917593.7253615, 0.2830437],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283592.6005045, 0.2494847],
            [7.7632705, 0.7690422, 1565265.2188750, 0.2597097],
            [7.3579086, 0.7805112, 1332252.5927338, 0.2657518],
        ],
    ]
)

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

    test_results = np.zeros((3, 3, n_turbines, 4))

    velocities = floris.flow_field.u[:, :, :, :, :]
    n_wind_directions = np.shape(velocities)[0]
    n_wind_speeds = np.shape(velocities)[1]

    yaw_angles = floris.farm.farm_controller.yaw_angles
    thrust_interpolation_func = np.array(n_wind_directions * n_wind_speeds * [t.fCt for t in turbines]).reshape(
        (n_wind_directions, n_wind_speeds, n_turbines)
    )
    power_interpolation_func = np.array(n_wind_directions * n_wind_speeds * [t.power_interp for t in turbines]).reshape(
        (n_wind_directions, n_wind_speeds, n_turbines)
    )
    power_exponent = np.array(n_wind_directions * n_wind_speeds * [t.pP for t in turbines]).reshape(
        (n_wind_directions, n_wind_speeds, n_turbines)
    )

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        thrust_interpolation_func,
    )
    farm_powers = power(
        np.array(n_turbines * n_wind_speeds * n_wind_directions * [floris.flow_field.air_density]).reshape(
            (n_wind_directions, n_wind_speeds, n_turbines)
        ),
        velocities,
        yaw_angles,
        power_exponent,
        power_interpolation_func,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        thrust_interpolation_func,
    )
    for i in range(n_wind_directions):
        for j in range(n_wind_speeds):
            for k in range(n_turbines):
                test_results[i, j, k, 0] = farm_avg_velocities[i, j, k]
                test_results[i, j, k, 1] = farm_cts[i, j, k]
                test_results[i, j, k, 2] = farm_powers[i, j, k]
                test_results[i, j, k, 3] = farm_axial_inductions[i, j, k]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
        )

    assert_results_arrays(test_results[0], baseline)


# def test_regression_rotation(sample_inputs_fixture):
#     """
#     Turbines in tandem and rotated.
#     The result from 270 degrees should match the results from 360 degrees.
#     """
#     sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
#     sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL
#     floris = Floris(input_dict=sample_inputs_fixture.floris)
#     fresh_turbine = copy.deepcopy(floris.farm.turbine_map.turbines[0])
#     wind_map = floris.farm.wind_map

#     # Generate the unrotated baselines
#     floris.farm.flow_field.calculate_wake()
#     [unwaked_baseline, first_waked_baseline, second_waked_baseline] = turbines_to_array(
#         floris.farm.turbine_map.turbines
#     )

#     # Rotate and calculate
#     wind_map.input_direction = [360]
#     wind_map.calculate_wind_direction()
#     new_map = TurbineMap(
#         [0.0, 0.0, 0.0],
#         [
#             10 * sample_inputs_fixture.floris["turbine"]["rotor_diameter"],
#             5 * sample_inputs_fixture.floris["turbine"]["rotor_diameter"],
#             0.0,
#         ],
#         [
#             copy.deepcopy(fresh_turbine),
#             copy.deepcopy(fresh_turbine),
#             copy.deepcopy(fresh_turbine),
#         ],
#     )
#     floris.farm.flow_field.reinitialize_flow_field(turbine_map=new_map, wind_map=wind_map)
#     floris.farm.flow_field.calculate_wake()

#     # Compare results to baaselines
#     test_results = turbines_to_array(floris.farm.turbine_map.turbines)
#     assert test_results[0] == approx(unwaked_baseline)
#     assert test_results[1] == approx(first_waked_baseline)
#     assert test_results[2] == approx(second_waked_baseline)


def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    sample_inputs_fixture.floris["wake"]["properties"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["properties"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris(input_dict=sample_inputs_fixture.floris)

    # yaw the upstream turbine 5 degrees
    floris.farm.farm_controller.set_yaw_angles([5.0, 0.0, 0.0])

    floris.go()

    turbines = floris.farm.turbines
    n_turbines = len(turbines)

    test_results = np.zeros((3, 3, n_turbines, 4))

    velocities = floris.flow_field.u[:, :, :, :, :]
    n_wind_directions = np.shape(velocities)[0]
    n_wind_speeds = np.shape(velocities)[1]

    yaw_angles = floris.farm.farm_controller.yaw_angles
    thrust_interpolation_func = np.array(n_wind_directions * n_wind_speeds * [t.fCt for t in turbines]).reshape(
        (n_wind_directions, n_wind_speeds, n_turbines)
    )
    power_interpolation_func = np.array(n_wind_directions * n_wind_speeds * [t.power_interp for t in turbines]).reshape(
        (n_wind_directions, n_wind_speeds, n_turbines)
    )
    power_exponent = np.array(n_wind_directions * n_wind_speeds * [t.pP for t in turbines]).reshape(
        (n_wind_directions, n_wind_speeds, n_turbines)
    )

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        thrust_interpolation_func,
    )
    farm_powers = power(
        np.array(n_turbines * n_wind_speeds * n_wind_directions * [floris.flow_field.air_density]).reshape(
            (n_wind_directions, n_wind_speeds, n_turbines)
        ),
        velocities,
        yaw_angles,
        power_exponent,
        power_interpolation_func,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        thrust_interpolation_func,
    )
    for i in range(n_wind_directions):
        for j in range(n_wind_speeds):
            for k in range(n_turbines):
                test_results[i, j, k, 0] = farm_avg_velocities[i, j, k]
                test_results[i, j, k, 1] = farm_cts[i, j, k]
                test_results[i, j, k, 2] = farm_powers[i, j, k]
                test_results[i, j, k, 3] = farm_axial_inductions[i, j, k]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
        )

    assert_results_arrays(test_results[0], yawed_baseline)
