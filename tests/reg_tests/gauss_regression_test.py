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

import numpy as np

from floris.simulation import Floris
from floris.simulation import Ct, power, axial_induction, average_velocity
from tests.conftest import N_TURBINES, N_WIND_DIRECTIONS, N_WIND_SPEEDS, print_test_values, assert_results_arrays

DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7634300, 1695368.6455473, 0.2568077],
            [5.8384411, 0.8436903, 651362.9121753, 0.3023199],
            [5.9388958, 0.8385498, 686209.4710003, 0.2990957],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7625731, 2413659.0651694, 0.2563676],
            [6.5698070, 0.8095679, 942487.3932503, 0.2818073],
            [6.7192788, 0.8035535, 1012058.4081816, 0.2783886],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7527803, 3306006.9741814, 0.2513940],
            [7.3198945, 0.7817588, 1312121.9341194, 0.2664185],
            [7.4982017, 0.7759067, 1406546.0953528, 0.2633075],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7304328, 4373591.7174990, 0.2404007],
            [ 8.1044931, 0.7626381, 1778225.5062060, 0.2564010],
            [ 8.2645633, 0.7622021, 1887139.2890270, 0.2561774],
        ]
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.3885389, 0.2548147],
            [5.8728701, 0.8419285, 663305.9063892, 0.3012090],
            [5.9429700, 0.8383413, 687622.7755572, 0.2989660],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397237.3791443, 0.2543815],
            [6.6084797, 0.8080118, 960487.4337100, 0.2809177],
            [6.7240659, 0.8033608, 1014286.5451750, 0.2782799],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283592.6005045, 0.2494847],
            [7.3621010, 0.7803736, 1334472.7586013, 0.2656784],
            [7.5035925, 0.7757548, 1409651.2478433, 0.2632273],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344217.6993801, 0.2386508],
            [8.1489867, 0.7625169, 1808499.5183449, 0.2563388],
            [8.2684171, 0.7621916, 1889761.4847929, 0.2561720],
        ]
    ]
)

# gch_baseline = [
#     (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
#     (0.8392733, 720745.6467388, 0.2995463, 5.8888694),
#     (0.8354140, 755306.6488337, 0.2971540, 5.9719556),
# ]

# yaw_added_recovery_baseline = [
#     (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
#     (0.8392735, 720743.5433910, 0.2995465, 5.8888643),
#     (0.8356813, 752912.7537170, 0.2973188, 5.9662006),
# ]

# secondary_steering_baseline = [
#     (0.7626396, 1802595.0749161, 0.2558908, 7.9803783),
#     (0.8401691, 712723.9446334, 0.3001057, 5.8695848),
#     (0.8363526, 746901.3150465, 0.2977332, 5.9517488),
# ]

# Note: compare the yawed vs non-yawed results. The upstream turbine
# power should be lower in the yawed case. The following turbine
# powers should higher in the yawed case.


def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris(sample_inputs_fixture.floris)
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_wind_speeds = floris.flow_field.n_wind_speeds
    n_wind_directions = floris.flow_field.n_wind_directions

    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles
    test_results = np.zeros((n_wind_directions, n_wind_speeds, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        floris.farm.fCt_interp,
    )
    farm_powers = power(
        np.array(n_turbines * n_wind_speeds * n_wind_directions * [floris.flow_field.air_density]).reshape(
            (n_wind_directions, n_wind_speeds, n_turbines)
        ),
        velocities,
        yaw_angles,
        floris.farm.pP,
        floris.farm.power_interp,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        floris.farm.fCt_interp,
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


def test_regression_rotation(sample_inputs_fixture):
    """
    Turbines in tandem and rotated.
    The result from 270 degrees should match the results from 360 degrees.

    Wind from the West (Left)
      0         1  x->
     |__________|
    0|0         2
     |
     |
     |
    1|1         3

    y
    |
    V

    Wind from the North (Top), rotated
      0         1  x->
     |__________|
    0|2         3
     |
     |
     |
    1|0         1

    y
    |
    V

    In 270, turbines 2 and 3 are waked. In 360, turbines 1 and 3 are waked.
    The test compares turbines 2 and 3 with 1 and 3 from 270 and 360.
    """
    TURBINE_DIAMETER = sample_inputs_fixture.floris["turbine"]["rotor_diameter"]

    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.floris["farm"]["layout_x"] = [
        0.0,
        0.0,
        5 * TURBINE_DIAMETER,
        5 * TURBINE_DIAMETER,
    ]
    sample_inputs_fixture.floris["farm"]["layout_y"] = [
        0.0,
        5 * TURBINE_DIAMETER,
        0.0,
        5 * TURBINE_DIAMETER
    ]
    sample_inputs_fixture.floris["flow_field"]["wind_directions"] = [270.0, 360.0]
    sample_inputs_fixture.floris["flow_field"]["wind_speeds"] = [8.0]

    floris = Floris(sample_inputs_fixture.floris)
    floris.steady_state_atmospheric_condition()

    velocities = floris.flow_field.u[:, :, :, :, :]

    farm_avg_velocities = average_velocity(
        velocities,
    )

    t0_270 = farm_avg_velocities[0, 0, 0]  # upstream
    t1_270 = farm_avg_velocities[0, 0, 1]  # upstream
    t2_270 = farm_avg_velocities[0, 0, 2]  # waked
    t3_270 = farm_avg_velocities[0, 0, 3]  # waked

    t0_360 = farm_avg_velocities[1, 0, 0]  # upstream
    t1_360 = farm_avg_velocities[1, 0, 1]  # waked
    t2_360 = farm_avg_velocities[1, 0, 2]  # upstream
    t3_360 = farm_avg_velocities[1, 0, 3]  # waked
    
    assert np.array_equal(t0_270, t2_360)
    assert np.array_equal(t1_270, t0_360)
    assert np.array_equal(t2_270, t3_360)
    assert np.array_equal(t3_270, t1_360)


def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris(sample_inputs_fixture.floris)

    yaw_angles = np.zeros((N_WIND_DIRECTIONS, N_WIND_SPEEDS, N_TURBINES))
    yaw_angles[:,:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_wind_speeds = floris.flow_field.n_wind_speeds
    n_wind_directions = floris.flow_field.n_wind_directions

    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles
    test_results = np.zeros((n_wind_directions, n_wind_speeds, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        floris.farm.fCt_interp,
    )
    farm_powers = power(
        np.array(n_turbines * n_wind_speeds * n_wind_directions * [floris.flow_field.air_density]).reshape(
            (n_wind_directions, n_wind_speeds, n_turbines)
        ),
        velocities,
        yaw_angles,
        floris.farm.pP,
        floris.farm.power_interp,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        floris.farm.fCt_interp,
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
