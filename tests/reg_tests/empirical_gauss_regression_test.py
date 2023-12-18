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
VELOCITY_MODEL = "empirical_gauss"
DEFLECTION_MODEL = "empirical_gauss"
TURBULENCE_MODEL = "wake_induced_mixing"


baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7636044, 1691326.6483808, 0.2568973],
            [5.8890878, 0.8410986, 668931.9953790, 0.3006878],
            [5.9448342, 0.8382459, 688269.8273350, 0.2989067],
        ],
        # 9m/s
        [
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
            [6.6288143, 0.8071935, 969952.7378773, 0.2804513],
            [6.7440713, 0.8025559, 1023598.6805729, 0.2778266],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
            [7.4019251, 0.7790665, 1355562.9527211, 0.2649822],
            [7.5493339, 0.7745724, 1437063.0620195, 0.2626039],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
            [8.2349756, 0.7622827, 1867008.5657835, 0.2562187],
            [8.3523516, 0.7619629, 1946873.1634864, 0.2560548],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.9257102, 0.8392246, 681635.9273649, 0.2995159],
            [5.9615388, 0.8373911, 694064.4542077, 0.2983761],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.6698959, 0.8055405, 989074.0018995, 0.2795122],
            [6.7631531, 0.8017881, 1032480.2286024, 0.2773950],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.4463751, 0.7776077, 1379101.8806016, 0.2642075],
            [7.5701211, 0.7740351, 1449519.8581580, 0.2623212],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.2809317, 0.7621575, 1898277.8462234, 0.2561545],
            [8.3710828, 0.7619119, 1959618.1795131, 0.2560286],
        ],
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.9343009, 0.8387850, 684615.9328740, 0.2992420],
            [5.9680241, 0.8370593, 696314.1525222, 0.2981704],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.6795240, 0.8051531, 993555.3595338, 0.2792927],
            [6.7704684, 0.8014937, 1035885.1172753, 0.2772298],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.4567077, 0.7772686, 1384573.5845651, 0.2640278],
            [7.5779862, 0.7738318, 1454233.0717541, 0.2622143],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.2914104, 0.7621290, 1905407.7287412, 0.2561399],
            [8.3784336, 0.7618919, 1964619.7950752, 0.2560184],
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
    sample_inputs_fixture.floris["wake"]["model_strings"]["turbulence_model"] = TURBULENCE_MODEL

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
    sample_inputs_fixture.floris["wake"]["model_strings"]["turbulence_model"] = TURBULENCE_MODEL
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
    sample_inputs_fixture.floris["wake"]["model_strings"]["turbulence_model"] = TURBULENCE_MODEL

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
            max_findex_print=4
        )

    assert_results_arrays(test_results[0:4], yawed_baseline)

def test_regression_yaw_added_recovery(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed and yaw added recovery
    correction enabled
    """

    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["turbulence_model"] = TURBULENCE_MODEL

    # Turn on yaw added recovery
    sample_inputs_fixture.floris["wake"]["enable_yaw_added_recovery"] = True
    # First pass, leave at default value of 0; should then do nothing

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

    # Compare to case where enable_yaw_added_recovery = False, since
    # default gains are 0.
    assert_results_arrays(test_results[0:4], yawed_baseline)

    # Second pass, use nonzero gain
    sample_inputs_fixture.floris["wake"]["wake_deflection_parameters"]\
        ["empirical_gauss"]["yaw_added_mixing_gain"] = 0.1

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
            max_findex_print=4
        )

    assert_results_arrays(test_results[0:4], yaw_added_recovery_baseline)


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
    sample_inputs_fixture.floris["wake"]["model_strings"]["turbulence_model"] = TURBULENCE_MODEL
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
    # Column 5 is completely unwaked in this model
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20])
    assert np.allclose(farm_powers[8,20], farm_powers[8,0])
    assert np.allclose(farm_powers[8,21], farm_powers[8,21:25])
