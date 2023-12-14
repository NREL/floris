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
VELOCITY_MODEL = "cc"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7636044, 1691326.6483808, 0.2568973],
            [5.4838164, 0.8620156, 529225.9172271, 0.3142687],
            [5.0221433, 0.8907283, 394126.6156555, 0.3347186],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
            [6.1712539, 0.8275295, 776795.0248898, 0.2923521],
            [5.6500663, 0.8533298, 586018.0719934, 0.3085123],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
            [6.8779113, 0.7971705, 1085894.0434488, 0.2748170],
            [6.2985764, 0.8216609, 828383.6208269, 0.2888489],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
            [7.6258784, 0.7725938, 1482932.7552807, 0.2615643],
            [6.9611771, 0.7938200, 1124649.7898263, 0.2729648],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.5274362, 0.8596051, 543479.0426304, 0.3126534],
            [5.0310723, 0.8901730, 396739.4832795, 0.3342992],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.2202711, 0.8252701, 796655.8471824, 0.2909965],
            [5.6617378, 0.8527326, 590066.7909898, 0.3081228],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [6.9317633, 0.7950036, 1110959.2451850, 0.2736173],
            [6.3125748, 0.8210156, 834055.5094286, 0.2884673],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [7.6832308, 0.7711112, 1517301.5142304, 0.2607884],
            [6.9761726, 0.7932167, 1131629.3899797, 0.2726328],
        ],
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.5431146, 0.8588028, 548917.6953551, 0.3121189],
            [5.0453462, 0.8892852, 400916.4566323, 0.3336309],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.2378520, 0.8244598, 803779.2831349, 0.2905124],
            [5.6785118, 0.8518742, 595885.4921489, 0.3075644],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [6.9507085, 0.7942413, 1119777.2268361, 0.2731968],
            [6.3312183, 0.8201563, 841609.4907163, 0.2879601],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [7.7025449, 0.7706119, 1528875.6023356, 0.2605276],
            [6.9954994, 0.7924390, 1140624.9700319, 0.2722057],
        ],
    ]
)

secondary_steering_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.5274367, 0.8596051, 543479.2092235, 0.3126534],
            [5.0364358, 0.8898394, 398309.0269631, 0.3340477],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.2202717, 0.8252701, 796656.0654567, 0.2909965],
            [5.6680298, 0.8524106, 592249.4291781, 0.3079132],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [6.9317639, 0.7950036, 1110959.5162103, 0.2736173],
            [6.3196140, 0.8206912, 836907.6633514, 0.2882756],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [7.6832314, 0.7711112, 1517301.8723625, 0.2607884],
            [6.9837299, 0.7929126, 1135146.9152189, 0.2724657],
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
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        floris.farm.ref_tilt_cp_cts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.farm.ref_density_cp_cts,
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        floris.farm.ref_tilt_cp_cts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.farm.ref_density_cp_cts,
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        )

    assert_results_arrays(test_results[0:4], yawed_baseline)


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
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        floris.farm.ref_tilt_cp_cts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.farm.ref_density_cp_cts,
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        )

    assert_results_arrays(test_results[0:4], yaw_added_recovery_baseline)


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
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        floris.farm.ref_tilt_cp_cts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.farm.ref_density_cp_cts,
        farm_eff_velocities,
        floris.farm.turbine_power_interps,
        floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
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
        )

    assert_results_arrays(test_results[0:4], secondary_steering_baseline)


def test_regression_small_grid_rotation(sample_inputs_fixture):
    """
    This utilizes a 5x5 wind farm with the layout in a regular grid oriented along the cardinal
    directions. The wind direction in this test is from 285 degrees which is slightly north of
    west. The objective of this test is to create a case with a very slight rotation of the wind
    farm to target the rotation and masking routines.

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
    tilt_angles = floris.farm.tilt_angles

    farm_eff_velocities = rotor_effective_velocity(
        floris.flow_field.air_density,
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.ref_tilt_cp_cts,
        floris.farm.pPs,
        floris.farm.pTs,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_powers = power(
        floris.farm.ref_density_cp_cts,
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
