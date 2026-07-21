
import numpy as np

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
VELOCITY_MODEL = "turboparkgauss"
DEFLECTION_MODEL = "none"
COMBINATION_MODEL = "sosfs"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [5.3669227, 0.8968386, 526338.6265211, 0.3394063],
            [4.7093471, 0.9414651, 338146.6902995, 0.3790301],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [6.0385619, 0.8590958, 754925.9561188, 0.3123139],
            [5.1942704, 0.9066535, 468726.5982167, 0.3472367],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [6.7109723, 0.8285157, 1057233.8964038, 0.2929467],
            [5.7307698, 0.8761547, 647750.0520513, 0.3240417],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [7.4177796, 0.8004049, 1413470.6329668, 0.2766196],
            [6.3151278, 0.8465180, 879266.7640038, 0.3041161],
        ],
    ]
)

full_flow_baseline = np.array(
    [
        [
            [
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
            ],
            [
                [7.88772229, 7.99999863, 8.10178685],
                [7.79725047, 7.90606371, 8.00885965],
                [4.18190854, 4.15233328, 4.29539865],
                [7.79725047, 7.90606371, 8.00885965],
                [7.88772229, 7.99999863, 8.10178685],
            ],
            [
                [7.88768866, 7.99996389, 8.1017523],
                [7.66496443, 7.76984581, 7.87298101],
                [3.67167327, 3.64367424, 3.771272],
                [7.66496443, 7.76984581, 7.87298101],
                [7.88768866, 7.99996389, 8.1017523],
            ],
            [
                [7.88743244, 7.99970024, 8.10148911],
                [7.51202294, 7.61308124, 7.71586343],
                [3.59890431, 3.58223279, 3.69628297],
                [7.51202294, 7.61308124, 7.71586343],
                [7.88743244, 7.99970024, 8.10148911],
            ],
            [
                [7.88684976, 7.99910218, 8.1008904],
                [7.45425328, 7.55442766, 7.6564573],
                [4.22344363, 4.23243562, 4.33734116],
                [7.45425328, 7.55442766, 7.6564573],
                [7.88684976, 7.99910218, 8.1008904]
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
    sample_inputs_fixture.core["wake"]["model_strings"]["turbulence_model"] = "none"

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.solve_for_turbines()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        turbines=floris.farm.turbines,
        velocities=velocities,
        turbulence_intensities=turbulence_intensities,
        air_density=air_density,
        yaw_angles=yaw_angles,
        power_setpoints=power_setpoints,
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
        turbine_type_map=floris.farm.turbine_type_map,
    )
    farm_powers = power(
        turbines=floris.farm.turbines,
        velocities=velocities,
        turbulence_intensities=turbulence_intensities,
        air_density=air_density,
        yaw_angles=yaw_angles,
        power_setpoints=power_setpoints,
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
        turbine_type_map=floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        turbines=floris.farm.turbines,
        velocities=velocities,
        turbulence_intensities=turbulence_intensities,
        air_density=air_density,
        yaw_angles=yaw_angles,
        power_setpoints=power_setpoints,
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
        turbine_type_map=floris.farm.turbine_type_map,
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
            max_findex_print=4,
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
    sample_inputs_fixture.core["wake"]["model_strings"]["turbulence_model"] = "none"
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
    floris.solve_for_turbines()

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
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["turbulence_model"] = "none"
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
    floris.solve_for_turbines()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    yaw_angles = floris.farm.yaw_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers = power(
        turbines=floris.farm.turbines,
        velocities=velocities,
        turbulence_intensities=turbulence_intensities,
        air_density=floris.flow_field.air_density,
        yaw_angles=yaw_angles,
        power_setpoints=power_setpoints,
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
        turbine_type_map=floris.farm.turbine_type_map,
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

# TurboParkGauss enables full_flow_solver
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
    sample_inputs_fixture.core["wake"]["model_strings"]["turbulence_model"] = "none"
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
