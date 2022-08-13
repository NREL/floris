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
VELOCITY_MODEL = "cc"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7634300, 1695368.7987130, 0.2568077],
            [5.3585771, 0.8698045, 492577.1572448, 0.3195870],
            [4.9669368, 0.8944963, 379718.4177412, 0.3375933],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7625731, 2413658.0981405, 0.2563676],
            [6.0299374, 0.8340431, 719536.2896199, 0.2963110],
            [5.5776779, 0.8570341, 560907.3153174, 0.3109458],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7527803, 3306006.2306084, 0.2513940],
            [6.7205268, 0.8035032, 1012640.0064192, 0.2783602],
            [6.2110904, 0.8256933, 792936.0055473, 0.2912497],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7304328, 4373596.1594956, 0.2404007],
            [7.4516751, 0.7774337, 1381908.5141696, 0.2641154],
            [6.8606000, 0.7978670, 1077836.5443900, 0.2752040],
        ]
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.5765064, 0.2548147],
            [5.4029703, 0.8670436, 505567.9316776, 0.3176841],
            [4.9742156, 0.8939700, 381463.8369041, 0.3371887],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397236.5542849, 0.2543815],
            [6.0798421, 0.8317429, 739756.7356164, 0.2949042],
            [5.5879702, 0.8565074, 564477.6027506, 0.3105979],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283591.8023665, 0.2494847],
            [6.7754450, 0.8012935, 1038201.4571916, 0.2771174],
            [6.2236744, 0.8251133, 798034.8027193, 0.2909027],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344222.0129382, 0.2386508],
            [7.5103951, 0.7755790, 1413728.7289467, 0.2631345],
            [6.8746872, 0.7973002, 1084393.3749950, 0.2748890],
        ]
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.5765064, 0.2548147],
            [5.4219904, 0.8658607, 511133.7736997, 0.3168748],
            [4.9901603, 0.8928170, 385287.3116696, 0.3363059],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397236.5542849, 0.2543815],
            [6.1011855, 0.8307591, 748404.6404163, 0.2943055],
            [5.6071092, 0.8555280, 571116.7279097, 0.3099527],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283591.8023665, 0.2494847],
            [6.7984638, 0.8003672, 1048915.4794254, 0.2765986],
            [6.2451030, 0.8241256, 806717.2493019, 0.2903131],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344222.0129382, 0.2386508],
            [7.5339320, 0.7749706, 1427833.3888763, 0.2628137],
            [6.8970594, 0.7964000, 1094806.4414958, 0.2743897],
        ],
    ]
)

secondary_steering_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.5765064, 0.2548147],
            [5.4029709, 0.8670436, 505568.1176628, 0.3176840],
            [4.9790760, 0.8936185, 382629.3354701, 0.3369191],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397236.5542849, 0.2543815],
            [6.0798429, 0.8317428, 739757.0246720, 0.2949042],
            [5.5937356, 0.8562124, 566477.5644593, 0.3104033],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283591.8023665, 0.2494847],
            [6.7754458, 0.8012934, 1038201.8164555, 0.2771174],
            [6.2301672, 0.8248140, 800665.5335362, 0.2907239],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344222.0129382, 0.2386508],
            [7.5103959, 0.7755790, 1413729.2052485, 0.2631345],
            [6.8816977, 0.7970181, 1087656.4020125, 0.2747324],
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
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris.from_dict(sample_inputs_fixture.floris)
    floris.initialize_domain()
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
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.flow_field.air_density,
        velocities,
        yaw_angles,
        floris.farm.pPs,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
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

    ^
    |
    y

    1|1         3
     |
     |
     |
    0|0         2
     |----------|
      0         1  x->


    Wind from the North (Top), rotated

    ^
    |
    y

    1|3         2
     |
     |
     |
    0|1         0
     |----------|
      0         1  x->

    In 270, turbines 2 and 3 are waked. In 360, turbines 0 and 2 are waked.
    The test compares turbines 2 and 3 with 0 and 2 from 270 and 360.
    """
    TURBINE_DIAMETER = 126.0

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

    floris = Floris.from_dict(sample_inputs_fixture.floris)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    farm_avg_velocities = average_velocity(floris.flow_field.u)

    t0_270 = farm_avg_velocities[0, 0, 0]  # upstream
    t1_270 = farm_avg_velocities[0, 0, 1]  # upstream
    t2_270 = farm_avg_velocities[0, 0, 2]  # waked
    t3_270 = farm_avg_velocities[0, 0, 3]  # waked

    t0_360 = farm_avg_velocities[1, 0, 0]  # waked
    t1_360 = farm_avg_velocities[1, 0, 1]  # upstream
    t2_360 = farm_avg_velocities[1, 0, 2]  # waked
    t3_360 = farm_avg_velocities[1, 0, 3]  # upstream

    assert np.allclose(t0_270, t1_360)
    assert np.allclose(t1_270, t3_360)
    assert np.allclose(t2_270, t0_360)
    assert np.allclose(t3_270, t2_360)


def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Floris.from_dict(sample_inputs_fixture.floris)

    yaw_angles = np.zeros((N_WIND_DIRECTIONS, N_WIND_SPEEDS, N_TURBINES))
    yaw_angles[:,:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
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
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.flow_field.air_density,
        velocities,
        yaw_angles,
        floris.farm.pPs,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
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


def test_regression_yaw_added_recovery(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """

    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    sample_inputs_fixture.floris["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.floris["wake"]["enable_secondary_steering"] = False
    sample_inputs_fixture.floris["wake"]["enable_yaw_added_recovery"] = True

    floris = Floris.from_dict(sample_inputs_fixture.floris)

    yaw_angles = np.zeros((N_WIND_DIRECTIONS, N_WIND_SPEEDS, N_TURBINES))
    yaw_angles[:,:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles
    
    floris.initialize_domain()
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
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.flow_field.air_density,
        velocities,
        yaw_angles,
        floris.farm.pPs,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
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

    assert_results_arrays(test_results[0], yaw_added_recovery_baseline)


def test_regression_secondary_steering(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and secondary steering enabled
    """

    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    sample_inputs_fixture.floris["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.floris["wake"]["enable_secondary_steering"] = True
    sample_inputs_fixture.floris["wake"]["enable_yaw_added_recovery"] = False

    floris = Floris.from_dict(sample_inputs_fixture.floris)

    yaw_angles = np.zeros((N_WIND_DIRECTIONS, N_WIND_SPEEDS, N_TURBINES))
    yaw_angles[:,:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles
    
    floris.initialize_domain()
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
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.flow_field.air_density,
        velocities,
        yaw_angles,
        floris.farm.pPs,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        floris.farm.turbine_fCts,
        floris.farm.turbine_type_map,
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

    assert_results_arrays(test_results[0], secondary_steering_baseline)


def test_regression_small_grid_rotation(sample_inputs_fixture):
    """
    Where wake models are masked based on the x-location of a turbine, numerical precision
    can cause masking to fail unexpectedly. For example, in the configuration here one of
    the turbines has these delta x values;

    [[4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]]

    and therefore the masking statement is False when it should be True. This causes the current
    turbine to be affected by its own wake. This test requires that at least in this particular
    configuration the masking correctly filters grid points.
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    X, Y = np.meshgrid(
        6.0 * 126.0 * np.arange(0, 5, 1),
        6.0 * 126.0 * np.arange(0, 5, 1)
    )
    X = X.flatten()
    Y = Y.flatten()

    sample_inputs_fixture.floris["farm"]["layout_x"] = X
    sample_inputs_fixture.floris["farm"]["layout_y"] = Y

    floris = Floris.from_dict(sample_inputs_fixture.floris)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles

    farm_powers = power(
        floris.flow_field.air_density,
        velocities,
        yaw_angles,
        floris.farm.pPs,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 leading turbine is completely unwaked
    # and the rest of the turbines have a partial wake from their immediate upstream turbine
    assert np.allclose(farm_powers[2,0,0:5], farm_powers[2,0,5:10])
    assert np.allclose(farm_powers[2,0,0:5], farm_powers[2,0,10:15])
    assert np.allclose(farm_powers[2,0,0:5], farm_powers[2,0,15:20])
    assert np.allclose(farm_powers[2,0,20], farm_powers[2,0,0])
    assert np.allclose(farm_powers[2,0,21], farm_powers[2,0,21:25])
