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
    N_TURBINES,
    N_WIND_DIRECTIONS,
    N_WIND_SPEEDS,
    print_test_values,
)


DEBUG = False
VELOCITY_MODEL = "super_gaussian_vawt"
DEFLECTION_MODEL = "none"
TURBULENCE_MODEL = "none"


baseline = np.array(
    [
        # 8 m/s
        [
            [7.9808916, 0.6400000, 128284.1947176, 0.2000000],
            [5.7072555, 0.6400000, 47157.2775396, 0.2000000],
            [5.3899835, 0.6400000, 39671.9573422, 0.2000000],
        ],
        # 9 m/s
        [
            [8.9785030, 0.6400000, 182645.8538054, 0.2000000],
            [6.4206624, 0.6400000, 66928.1744359, 0.2000000],
            [6.0637314, 0.6400000, 56371.3864073, 0.2000000],
        ],
        # 10 m/s
        [
            [9.9761145, 0.6400000, 250533.3207157, 0.2000000],
            [7.1340694, 0.6400000, 91857.4256469, 0.2000000],
            [6.7374793, 0.6400000, 77466.6641405, 0.2000000],
        ],
        # 11 m/s
        [
            [10.9737259, 0.6400000, 333449.2621454, 0.2000000],
            [7.8474763, 0.6400000, 122218.0126134, 0.2000000],
            [7.4112273, 0.6400000, 102886.3004900, 0.2000000],
        ],
    ]
)


def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["deflection_model"] = \
        DEFLECTION_MODEL
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["turbulence_model"] = \
        TURBULENCE_MODEL

    floris = Floris.from_dict(sample_inputs_fixture.floris_vawt)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_wind_speeds = floris.flow_field.n_wind_speeds
    n_wind_directions = floris.flow_field.n_wind_directions

    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    ref_tilt_cp_cts = (
        np.ones((n_wind_directions, n_wind_speeds, n_turbines))
        * floris.farm.ref_tilt_cp_cts
    )
    test_results = np.zeros((n_wind_directions, n_wind_speeds, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_eff_velocities = rotor_effective_velocity(
        floris.flow_field.air_density,
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        ref_tilt_cp_cts,
        floris.farm.pPs,
        floris.farm.pTs,
        floris.farm.turbine_fTilts,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        tilt_angles,
        ref_tilt_cp_cts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_fTilts,
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
        ref_tilt_cp_cts,
        floris.farm.turbine_fCts,
        floris.farm.turbine_fTilts,
        floris.farm.correct_cp_ct_for_tilt,
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
    TURBINE_DIAMETER = 26.0

    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["deflection_model"] = \
        DEFLECTION_MODEL
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["turbulence_model"] = \
        TURBULENCE_MODEL
    sample_inputs_fixture.floris_vawt["farm"]["layout_x"] = [
        0.0,
        0.0,
        5 * TURBINE_DIAMETER,
        5 * TURBINE_DIAMETER,
    ]
    sample_inputs_fixture.floris_vawt["farm"]["layout_y"] = [
        0.0,
        5 * TURBINE_DIAMETER,
        0.0,
        5 * TURBINE_DIAMETER,
    ]
    sample_inputs_fixture.floris_vawt["flow_field"]["wind_directions"] = [270.0, 360.0]
    sample_inputs_fixture.floris_vawt["flow_field"]["wind_speeds"] = [8.0]

    floris = Floris.from_dict(sample_inputs_fixture.floris_vawt)
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
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["deflection_model"] = \
        DEFLECTION_MODEL
    sample_inputs_fixture.floris_vawt["wake"]["model_strings"]["turbulence_model"] = \
        TURBULENCE_MODEL
    X, Y = np.meshgrid(
        6.0 * 26.0 * np.arange(0, 5, 1),
        6.0 * 26.0 * np.arange(0, 5, 1),
    )
    X = X.flatten()
    Y = Y.flatten()

    sample_inputs_fixture.floris_vawt["farm"]["layout_x"] = X
    sample_inputs_fixture.floris_vawt["farm"]["layout_y"] = Y

    floris = Floris.from_dict(sample_inputs_fixture.floris_vawt)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    ref_tilt_cp_cts = np.ones((1, 1, len(X))) * floris.farm.ref_tilt_cp_cts

    farm_eff_velocities = rotor_effective_velocity(
        floris.flow_field.air_density,
        floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        ref_tilt_cp_cts,
        floris.farm.pPs,
        floris.farm.pTs,
        floris.farm.turbine_fTilts,
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
    assert np.allclose(farm_powers[2,0,0:5], farm_powers[2,0,5:10])
    assert np.allclose(farm_powers[2,0,0:5], farm_powers[2,0,10:15])
    assert np.allclose(farm_powers[2,0,0:5], farm_powers[2,0,15:20])
    assert np.allclose(farm_powers[2,0,20], farm_powers[2,0,0])
    assert np.allclose(farm_powers[2,0,21], farm_powers[2,0,21:25])
