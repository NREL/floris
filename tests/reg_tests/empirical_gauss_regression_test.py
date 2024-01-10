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
    Floris,
    power,
    rotor_effective_velocity,
    thrust_coefficient,
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
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [5.8181628, 0.8711866, 676912.0380737, 0.3205471],
            [5.8941747, 0.8668654, 702276.3178047, 0.3175620],
        ],
        # 9m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [6.5498312, 0.8358441, 984786.7218587, 0.2974192],
            [6.6883370, 0.8295451, 1047057.3206209, 0.2935691],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [7.2852518, 0.8049506, 1339238.8882972, 0.2791780],
            [7.4865891, 0.7981254, 1452997.4778680, 0.2753477],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [8.1286243, 0.7869622, 1867298.1260108, 0.2692199],
            [8.2872457, 0.7867578, 1985849.6635654, 0.2691092],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.8572213, 0.8689662, 689945.4020673, 0.3190070],
            [5.9122259, 0.8658393, 708299.7846078, 0.3168602],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.5936194, 0.8338527, 1004473.3935880, 0.2961941],
            [6.7089679, 0.8286068, 1056332.7378826, 0.2930017],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.3336404, 0.8032764, 1366138.4198352, 0.2782323],
            [7.5095680, 0.7973796, 1466340.6394405, 0.2749331],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.1779964, 0.7868986, 1904198.1536702, 0.2691855],
            [8.3074034, 0.7867318, 2000915.2988301, 0.2690952],
        ],
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.8665710, 0.8684347, 693065.2795916, 0.3186403],
            [5.9193499, 0.8654343, 710676.9807602, 0.3165840],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.6040901, 0.8333765, 1009180.8710828, 0.2959023],
            [6.7169991, 0.8282416, 1059943.4814040, 0.2927813],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.3451916, 0.8028791, 1372599.7339512, 0.2780085],
            [7.5184292, 0.7971003, 1471563.4898254, 0.2747780],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.1895130, 0.7868837, 1912805.5199083, 0.2691774],
            [8.3154794, 0.7867214, 2006951.2349727, 0.2690895],
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
    farm_cts = thrust_coefficient(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        floris.farm.yaw_angles,
        floris.farm.tilt_angles,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
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
    farm_cts = thrust_coefficient(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        floris.farm.yaw_angles,
        floris.farm.tilt_angles,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
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
    farm_cts = thrust_coefficient(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        floris.farm.yaw_angles,
        floris.farm.tilt_angles,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
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
    farm_cts = thrust_coefficient(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        floris.farm.yaw_angles,
        floris.farm.tilt_angles,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
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

    # farm_eff_velocities = rotor_effective_velocity(
    #     floris.flow_field.air_density,
    #     floris.farm.ref_air_densities,
    #     velocities,
    #     yaw_angles,
    #     tilt_angles,
    #     floris.farm.ref_tilts,
    #     floris.farm.pPs,
    #     floris.farm.pTs,
    #     floris.farm.turbine_tilt_interps,
    #     floris.farm.correct_cp_ct_for_tilt,
    #     floris.farm.turbine_type_map,
    # )
    farm_powers = power(
        velocities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        floris.farm.yaw_angles,
        floris.farm.tilt_angles,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 is completely unwaked in this model
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20])
    assert np.allclose(farm_powers[8,20], farm_powers[8,0])
    assert np.allclose(farm_powers[8,21], farm_powers[8,21:25])
