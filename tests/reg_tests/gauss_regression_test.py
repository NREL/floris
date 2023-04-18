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
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7636044, 1691326.6483808, 0.2568973],
            [5.9535039, 0.8378023, 691277.2666766, 0.2986311],
            [6.0197522, 0.8345126, 715409.4436445, 0.2965993],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
            [6.6995977, 0.8043454, 1002898.6210841, 0.2788357],
            [6.8102318, 0.7998937, 1054392.8363310, 0.2763338],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
            [7.4637061, 0.7770389, 1388279.6564701, 0.2639062],
            [7.5999706, 0.7732635, 1467407.3821931, 0.2619157],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
            [8.2622911, 0.7622083, 1885594.4958198, 0.2561805],
            [8.3719551, 0.7619095, 1960211.6949745, 0.2560274],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.9856445, 0.8361576, 702426.4817361, 0.2976127],
            [6.0238963, 0.8343216, 717088.5782753, 0.2964819],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.7356851, 0.8028933, 1019695.3621240, 0.2780165],
            [6.8150684, 0.7996991, 1056644.0444495, 0.2762251],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.5030787, 0.7757681, 1409344.3206494, 0.2632343],
            [7.6053686, 0.7731239, 1470642.1508821, 0.2618425],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.3037405, 0.7620954, 1913797.3425937, 0.2561227],
            [8.3759415, 0.7618987, 1962924.0966747, 0.2560219],
        ],
    ]
)

"""
# These are the results from v2.4 develop branch
gch_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.3885389, 0.2548147],
            [5.8920347, 0.8409478, 669953.8921404, 0.3005933],
            [5.9690770, 0.8370054, 696678.9863587, 0.2981370],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397237.3791443, 0.2543815],
            [6.6299831, 0.8071465, 970496.1338006, 0.2804246],
            [6.7527627, 0.8022061, 1027643.3724351, 0.2776299],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283592.6005045, 0.2494847],
            [7.3852773, 0.7796129, 1346745.9407360, 0.2652730],
            [7.5343901, 0.7749587, 1428106.9252795, 0.2628074],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344217.6993801, 0.2386508],
            [8.1727131, 0.7624523, 1824643.2726943, 0.2563057],
            [8.2996789, 0.7621064, 1911032.3885037, 0.2561283],
        ]
    ]
)

secondary_steering_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.3885389, 0.2548147],
            [5.8728752, 0.8419282, 663307.6815433, 0.3012088],
            [5.9488299, 0.8380415, 689655.4839532, 0.2987797],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397237.3791443, 0.2543815],
            [6.6084854, 0.8080115, 960490.1060497, 0.2809176],
            [6.7305708, 0.8030991, 1017314.2281904, 0.2781324],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283592.6005045, 0.2494847],
            [7.3621072, 0.7803734, 1334476.0326665, 0.2656783],
            [7.5106613, 0.7755721, 1413887.2753700, 0.2631309],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344217.6993801, 0.2386508],
            [8.1489930, 0.7625169, 1808503.8150366, 0.2563388],
            [8.2759469, 0.7621711, 1894884.8361479, 0.2561615],
        ]
    ]
)
"""


gch_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [6.0012497, 0.8353654, 707912.6031236, 0.2971241],
            [6.0458168, 0.8333112, 725970.3069204, 0.2958623],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.7531826, 0.8021893, 1027839.4859975, 0.2776204],
            [6.8391301, 0.7987309, 1067843.4584263, 0.2756849],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.5219279, 0.7752809, 1420639.8615893, 0.2629772],
            [7.6309661, 0.7724622, 1485981.5768983, 0.2614954],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.3229930, 0.7620429, 1926897.0262401, 0.2560958],
            [8.4021717, 0.7618272, 1980771.5704442, 0.2559853],
        ],
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [6.0012490, 0.8353654, 707912.3201655, 0.2971241],
            [6.0404040, 0.8335607, 723777.1688957, 0.2960151],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.7531818, 0.8021893, 1027839.1215598, 0.2776204],
            [6.8331381, 0.7989720, 1065054.4872236, 0.2758193],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.5219271, 0.7752809, 1420639.3564230, 0.2629773],
            [7.6244680, 0.7726302, 1482087.5389477, 0.2615835],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.3229921, 0.7620429, 1926896.4413586, 0.2560958],
            [8.3952439, 0.7618461, 1976057.7564083, 0.2559949],
        ],
    ]
)

secondary_steering_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736330, 0.7606986, 1679924.0721706, 0.2549029],
            [5.9856452, 0.8361576, 702426.7279908, 0.2976127],
            [6.0294010, 0.8340678, 719318.9574833, 0.2963261],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7596552, 2391434.0080674, 0.2543734],
            [6.7356859, 0.8028933, 1019695.7325708, 0.2780165],
            [6.8211610, 0.7994540, 1059479.8255425, 0.2760882],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7500732, 3275671.6727516, 0.2495630],
            [7.5030795, 0.7757681, 1409344.8339510, 0.2632343],
            [7.6119726, 0.7729532, 1474599.5989813, 0.2617529],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7278454, 4333842.6695283, 0.2387424],
            [8.3037414, 0.7620954, 1913797.9363787, 0.2561227],
            [8.3829757, 0.7618795, 1967710.2678086, 0.2560120],
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


def test_regression_gch(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed, yaw added recovery
    correction enabled, and secondary steering enabled
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    ### With GCH off (via conftest), GCH should be same as Gauss

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

    # Don't use the test values here, gch is off! See the docstring.
    # if DEBUG:
    #     print_test_values(
    #         farm_avg_velocities,
    #         farm_cts,
    #         farm_powers,
    #         farm_axial_inductions,
    #     )

    assert_results_arrays(test_results[0], yawed_baseline)


    ### With GCH on, the results should change
    sample_inputs_fixture.floris["wake"]["enable_transverse_velocities"] = True
    sample_inputs_fixture.floris["wake"]["enable_secondary_steering"] = True
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

    assert_results_arrays(test_results[0], gch_baseline)


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
