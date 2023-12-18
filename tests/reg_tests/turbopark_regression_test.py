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
    rotor_effective_velocity,
)
from tests.conftest import (
    assert_results_arrays,
    N_FINDEX,
    N_TURBINES,
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
            [7.9736330, 0.9802361, 1755704.3055424, 0.4297079],
            [6.7151843, 0.9999000, 1059146.8565455, 0.4950000],
            [6.8184778, 0.9999000, 1105587.6223554, 0.4950000],
        ],
        # 9 m/s
        [
            [8.9703371, 0.9375172, 2496426.9963900, 0.3750172],
            [7.0397157, 0.9999000, 1210390.0246292, 0.4950000],
            [7.2470207, 0.9999000, 1331435.3804590, 0.4950000],
        ],
        # 10 m/s
        [
            [9.9670412, 0.9064422, 3417764.6183773, 0.3470639],
            [7.6483624, 0.9999000, 1565778.8192291, 0.4950000],
            [7.9169796, 0.9851761, 1722624.3881717, 0.4391232],
        ],
        # 11 m/s
        [
            [10.9637454, 0.8574973, 4519330.7327692, 0.3112524],
            [8.2847975, 0.9660736, 1983986.1649899, 0.4079044],
            [7.8740779, 0.9889170, 1697574.0914617, 0.4473620],
        ],
    ]
)


yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.9765060, 1744593.9143866, 0.4191177],
            [6.6588666, 0.9999000, 1033826.4024924, 0.4950000],
            [6.7775548, 0.9999000, 1087188.6519965, 0.4950000],
        ],
        # 9 m/s
        [
            [8.9703371, 0.9339496, 2480425.7212356, 0.3694929],
            [7.0359610, 0.9999000, 1208197.6518290, 0.4950000],
            [7.2560848, 0.9999000, 1336727.8980709, 0.4950000],
        ],
        # 10 m/s
        [
            [9.9670412, 0.9029930, 3395649.2030651, 0.3428407],
            [7.6585724, 0.9999000, 1571740.4125599, 0.4950000],
            [7.9402480, 0.9831471, 1736210.8307124, 0.4350907],
        ],
        # 11 m/s
        [
            [10.9637454, 0.8542343, 4488170.1670616, 0.3081595],
            [8.3069368, 0.9651514, 2000535.2514727, 0.4066611],
            [7.8881534, 0.9876896, 1705792.7650019, 0.4445239],
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
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_eff_velocities = rotor_effective_velocity(
        floris.flow_field.air_density,
        floris.farm.ref_air_densities,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.pPs,
        floris.farm.pTs,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4
        )

    assert_results_arrays(test_results[0:4], baseline)


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

    t0_270 = farm_avg_velocities[0, 0]  # upstream
    t1_270 = farm_avg_velocities[0, 1]  # upstream
    t2_270 = farm_avg_velocities[0, 2]  # waked
    t3_270 = farm_avg_velocities[0, 3]  # waked

    t0_360 = farm_avg_velocities[1, 0]  # waked
    t1_360 = farm_avg_velocities[1, 1]  # upstream
    t2_360 = farm_avg_velocities[1, 2]  # waked
    t3_360 = farm_avg_velocities[1, 3]  # upstream

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

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_eff_velocities = rotor_effective_velocity(
        floris.flow_field.air_density,
        floris.farm.ref_air_densities,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.pPs,
        floris.farm.pTs,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4
        )

    assert_results_arrays(test_results[0:4], yawed_baseline)

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
    tilt_angles = floris.farm.tilt_angles

    farm_eff_velocities = rotor_effective_velocity(
        floris.flow_field.air_density,
        floris.farm.ref_air_densities,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilts,
        floris.farm.pPs,
        floris.farm.pTs,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 leading turbine is completely unwaked
    # and the rest of the turbines have a partial wake from their immediate upstream turbine
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20])
    assert np.allclose(farm_powers[8,20], farm_powers[8,0])
    assert np.allclose(farm_powers[8,21], farm_powers[8,21:25])
