# Copyright 2022 NREL

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

from floris.simulation import (
    average_velocity,
    axial_induction,
    Ct,
    Floris,
    power,
)
from tests.conftest import (
    assert_results_arrays,
    N_TURBINES,
    N_WIND_DIRECTIONS,
    N_WIND_SPEEDS,
    print_test_values,
)


DEBUG = False
VELOCITY_MODEL = "turbopark"
DEFLECTION_MODEL = "gauss"
COMBINATION_MODEL = "fls"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7636044, 1691326.6483808, 0.2568973],
            [6.0583922, 0.8327316, 731065.6226282, 0.2955077],
            [5.4067009, 0.8668116, 506659.6232808, 0.3175251],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
            [6.8171892, 0.7996138, 1057631.1392858, 0.2761774],
            [6.0917181, 0.8311955, 744568.6379292, 0.2945709],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
            [7.5908545, 0.7734991, 1461944.4626519, 0.2620395],
            [6.7995666, 0.8003229, 1049428.7626183, 0.2765738],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
            [8.3975139, 0.7618399, 1977602.3128807, 0.2559918],
            [7.5196816, 0.7753389, 1419293.7479312, 0.2630079],
        ],
    ]
)


yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [6.0772917, 0.8318604, 738723.3410291, 0.2949759],
            [5.4215054, 0.8658908, 510991.8557577, 0.3168954],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.8384389, 0.7987587, 1067521.7514783, 0.2757004],
            [6.1089600, 0.8304008, 751554.7217137, 0.2940879],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.6142906, 0.7728933, 1475988.7044752, 0.2617214],
            [6.8186733, 0.7995541, 1058321.9413265, 0.2761440],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.4226213, 0.7617715, 1994685.7970084, 0.2559567],
            [7.5392355, 0.7748335, 1431011.5054545, 0.2627414],
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
    sample_inputs_fixture.floris["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL

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
        floris.farm.ref_density_cp_cts,
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
    sample_inputs_fixture.floris["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
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
        floris.farm.ref_density_cp_cts,
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
    sample_inputs_fixture.floris["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
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
        floris.farm.ref_density_cp_cts,
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
