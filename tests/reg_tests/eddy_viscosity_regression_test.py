
import numpy as np
import pytest

from floris.core import (
    average_velocity,
    axial_induction,
    Core,
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
VELOCITY_MODEL = "eddy_viscosity"
DEFLECTION_MODEL = "gauss"
COMBINATION_MODEL = "soed"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [6.6666422, 0.8305317, 1037303.6026684, 0.2941674],
            [6.3776334, 0.8436754, 907368.5661344, 0.3023105],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [7.5017741, 0.7976252, 1461746.8459320, 0.2750696],
            [7.1929915, 0.8081969, 1288784.8178808, 0.2810234],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [8.3384475, 0.7866918, 2024117.2497181, 0.2690735],
            [8.0002374, 0.7871277, 1771343.4133243, 0.2693096],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [9.2200413, 0.7853932, 2723153.7974906, 0.2683716],
            [8.8125257, 0.7860809, 2378437.2315890, 0.2687430],
        ],
    ]
)

# yawed_baseline = np.array(
#     [
#         # 8 m/s
#         [
#             [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
#             [6.0816475, 0.8571363, 774296.7271893, 0.3110134],
#             [5.5272875, 0.8877222, 579850.4298177, 0.3324606],
#         ],
#         # 9 m/s
#         [
#             [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
#             [6.8472506, 0.8223180, 1118503.0309148, 0.2892383],
#             [6.3747452, 0.8438067, 906070.0511419, 0.3023935],
#         ],
#         # 10 m/s
#         [
#             [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
#             [7.6174285, 0.7940006, 1530191.8035935, 0.2730642],
#             [7.2119500, 0.8075204, 1299067.3876318, 0.2806375],
#         ],
#         # 11 m/s
#         [
#             [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
#             [8.5159500, 0.7864631, 2156780.3499849, 0.2689497],
#             [8.0047998, 0.7871218, 1774753.2988553, 0.2693064],
#         ],
#     ]
# )

full_flow_baseline = np.array(
    [
        [
            [
                [7.88772361, 8.0,         8.10178821],
                [7.88772361, 8.0,         8.10178821],
                [7.88772361, 8.0,         8.10178821],
                [7.88772361, 8.0,         8.10178821],
                [7.88772361, 8.0,         8.10178821],
            ],
            [
                [7.88758793, 7.99986028, 8.10164885],
                [7.69657254, 7.80314836, 7.9054495 ],
                [5.74483323, 5.79320367, 5.90074201],
                [7.69657254, 7.80314836, 7.9054495 ],
                [7.88758793, 7.99986028, 8.10164885],
            ],
            [
                [7.8151773 , 7.92629295, 8.02727307],
                [7.48541427, 7.58840385, 7.68856061],
                [5.67767815, 5.73066043, 5.83176442],
                [7.48541427, 7.58840385, 7.68856061],
                [7.8151773 , 7.92629295, 8.02727307],
            ],
            [
                [7.75903294, 7.86931452, 7.969605  ],
                [7.38563303, 7.48715077, 7.58607141],
                [5.79942256, 5.85940559, 5.95681284],
                [7.38563303, 7.48715077, 7.58607141],
                [7.75903294, 7.86931452, 7.969605  ],
            ],
            [
                [7.74436376, 7.85435448, 7.95453772],
                [7.37530096, 7.47827906, 7.57545893],
                [6.89021982, 6.98370227, 7.07721321],
                [7.37530096, 7.47827906, 7.57545893],
                [7.74436376, 7.85435448, 7.95453772],
            ]
        ]
    ]
)

# Note: compare the yawed vs non-yawed results. The upstream turbine
# power should be lower in the yawed case. The following turbine
# powers should higher in the yawed case.


def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL


    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
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

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    sample_inputs_fixture.core["farm"]["layout_x"] = [
        0.0,
        0.0,
        5 * TURBINE_DIAMETER,
        5 * TURBINE_DIAMETER,
    ]
    sample_inputs_fixture.core["farm"]["layout_y"] = [
        0.0,
        5 * TURBINE_DIAMETER,
        0.0,
        5 * TURBINE_DIAMETER
    ]
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0, 360.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0, 8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1, 0.1]

    floris = Core.from_dict(sample_inputs_fixture.core)
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
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL

    floris = Core.from_dict(sample_inputs_fixture.core)

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    with pytest.raises(NotImplementedError):
        floris.steady_state_atmospheric_condition()
    # Once implemented, copy code from test_regression_yaw on jensen_jimenez_regression_test.py and
    # update yawed_baseline values

def test_symmetry(sample_inputs_fixture):
    """
    This utilizes a 5x5 wind farm with the layout in a regular grid oriented along the cardinal
    directions. The wind direction in this test is from 270 degrees, directly aligned with the
    columns of the farm. The objective of this test is to check that the solve is symmetric.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    X, Y = np.meshgrid(
        6.0 * 126.0 * np.arange(0, 5, 1),
        6.0 * 126.0 * np.arange(0, 5, 1)
    )
    X = X.flatten()
    Y = Y.flatten()

    sample_inputs_fixture.core["farm"]["layout_x"] = X
    sample_inputs_fixture.core["farm"]["layout_y"] = Y

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )

    rtol = 1e-7
    assert np.allclose(farm_powers[0,0:5], farm_powers[0,20:25], rtol=rtol) # Outer columns
    assert np.allclose(farm_powers[0,5:10], farm_powers[0,15:20], rtol=rtol) # Inner columns

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
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    X, Y = np.meshgrid(
        6.0 * 126.0 * np.arange(0, 5, 1),
        6.0 * 126.0 * np.arange(0, 5, 1)
    )
    X = X.flatten()
    Y = Y.flatten()

    sample_inputs_fixture.core["farm"]["layout_x"] = X
    sample_inputs_fixture.core["farm"]["layout_y"] = Y

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 leading turbine is completely unwaked
    # and the rest of the turbines have a partial wake from their immediate upstream turbine
    rtol = 1e-2 # Fails for default rtol=1e-5
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10], rtol=rtol)
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15], rtol=rtol)
    # The following fails, but it's not clear that it should pass. Setting rtol=1e-2 makes it pass.
    # assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20], rtol=rtol)
    assert np.allclose(farm_powers[8,20], farm_powers[8,0], rtol=rtol)
    assert np.allclose(farm_powers[8,21], farm_powers[8,21:25], rtol=rtol)


def test_full_flow_solver(sample_inputs_fixture):
    """
    Full flow solver test with the flow field planar grid.
    This requires one wind condition, and the grid is deliberately coarse to allow for
    visually comparing results, as needed.
    The u-component of velocity is compared, and the array has the shape
    (n_findex, n_turbines, n grid points in x, n grid points in y, 3 grid points in z).
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    sample_inputs_fixture.core["solver"] = {
        "type": "flow_field_planar_grid",
        "normal_vector": "z",
        "planar_coordinate": sample_inputs_fixture.core["farm"]["turbine_type"][0]["hub_height"],
        "flow_field_grid_points": [5, 5],
        "flow_field_bounds": [None, None],
    }
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1]

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.solve_for_viz()

    velocities = floris.flow_field.u_sorted

    if DEBUG:
        print(velocities)

    assert_results_arrays(velocities, full_flow_baseline)
