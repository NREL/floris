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
            [7.9803783, 0.7634300, 1695368.7987130, 0.2568077],
            [5.8384411, 0.8436903, 651363.2435524, 0.3023199],
            [5.9388958, 0.8385498, 686209.8630205, 0.2990957],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7625731, 2413658.0981405, 0.2563676],
            [6.5698070, 0.8095679, 942487.9831258, 0.2818073],
            [6.7192788, 0.8035535, 1012059.0934624, 0.2783886],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7527803, 3306006.2306084, 0.2513940],
            [7.3198945, 0.7817588, 1312122.9051486, 0.2664185],
            [7.4982017, 0.7759067, 1406547.1257826, 0.2633075],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7304328, 4373596.1594956, 0.2404007],
            [8.1044931, 0.7626381, 1778226.0596889, 0.2564010],
            [8.2645633, 0.7622021, 1887140.5106744, 0.2561774],
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
            [7.9803783, 0.7605249, 1683956.5765064, 0.2548147],
            [5.8919486, 0.8409522, 669924.4096484, 0.3005960],
            [5.9686695, 0.8370262, 696538.0378027, 0.2981500],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397236.5542849, 0.2543815],
            [6.6298866, 0.8071504, 970451.8269047, 0.2804268],
            [6.7523126, 0.8022243, 1027434.5597156, 0.2776401],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283591.8023665, 0.2494847],
            [7.3851732, 0.7796164, 1346691.8170923, 0.2652748],
            [7.5339044, 0.7749713, 1427816.8489148, 0.2628140],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344222.0129382, 0.2386508],
            [8.1726065, 0.7624526, 1824571.5626205, 0.2563058],
            [8.2991708, 0.7621078, 1910688.0574225, 0.2561290],
        ],
    ]
)

yaw_added_recovery_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.5765064, 0.2548147],
            [5.8919476, 0.8409523, 669924.0609678, 0.3005961],
            [5.9630522, 0.8373137, 694589.4363406, 0.2983281],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397236.5542849, 0.2543815],
            [6.6298855, 0.8071504, 970451.3019789, 0.2804268],
            [6.7460763, 0.8024752, 1024531.8988965, 0.2777812],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283591.8023665, 0.2494847],
            [7.3851720, 0.7796164, 1346691.1737223, 0.2652748],
            [7.5271249, 0.7751465, 1423754.1608641, 0.2629064],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344222.0129382, 0.2386508],
            [8.1726052, 0.7624526, 1824570.7175565, 0.2563058],
            [8.2919410, 0.7621275, 1905768.7628771, 0.2561391],
        ],
    ]
)

secondary_steering_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7605249, 1683956.5765064, 0.2548147],
            [5.8728728, 0.8419284, 663307.1901296, 0.3012089],
            [5.9486952, 0.8380484, 689609.1551620, 0.2987839],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7596713, 2397236.5542849, 0.2543815],
            [6.6084827, 0.8080116, 960489.4504135, 0.2809176],
            [6.7304206, 0.8031051, 1017245.0103229, 0.2781358],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7499157, 3283591.8023665, 0.2494847],
            [7.3621043, 0.7803735, 1334475.4570600, 0.2656784],
            [7.5104978, 0.7755763, 1413790.2904370, 0.2631331],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7276532, 4344222.0129382, 0.2386508],
            [8.1489900, 0.7625169, 1808502.4860052, 0.2563388],
            [8.2757728, 0.7621716, 1894767.6143032, 0.2561617],
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
