# Copyright 2023 NREL

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
    power,
)
from floris.simulation.turbine import rotor_effective_velocity
from floris.tools import FlorisInterface
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
            [7.9736330, 0.7636044, 1691326.6483808, 0.2568973],
            [7.9736330, 0.7636044, 1691326.6483808, 0.2568973],
        ],
        # 9 m/s
        [
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
            [8.9703371, 0.7625570, 2407841.6718785, 0.2563594],
        ],
        # 10 m/s
        [
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
            [9.9670412, 0.7529384, 3298067.1555604, 0.2514735],
        ],
        # 11 m/s
        [
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
            [10.9637454, 0.7306256, 4363191.9880631, 0.2404936],
        ],
    ]
)


def test_calculate_no_wake(sample_inputs_fixture):
    """
    The calculate_no_wake function calculates the power production of a wind farm
    assuming no wake losses. It does this by initializing and finalizing the
    floris simulation while skipping the wake calculation. The power for all wind
    turbines should be the same for a uniform wind condition. The chosen wake model
    is not important since it will not actually be used. However, it is left enabled
    instead of using "None" so that additional tests can be constructed here such
    as one with yaw activated.
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fi = FlorisInterface(sample_inputs_fixture.floris)
    fi.calculate_no_wake()

    n_turbines = fi.floris.farm.n_turbines
    n_wind_speeds = fi.floris.flow_field.n_wind_speeds
    n_wind_directions = fi.floris.flow_field.n_wind_directions

    velocities = fi.floris.flow_field.u
    yaw_angles = fi.floris.farm.yaw_angles
    tilt_angles = fi.floris.farm.tilt_angles
    ref_tilt_cp_cts = (
        np.ones((n_wind_directions, n_wind_speeds, n_turbines))
        * fi.floris.farm.ref_tilt_cp_cts
    )
    test_results = np.zeros((n_wind_directions, n_wind_speeds, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_eff_velocities = rotor_effective_velocity(
        fi.floris.flow_field.air_density,
        fi.floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        tilt_angles,
        ref_tilt_cp_cts,
        fi.floris.farm.pPs,
        fi.floris.farm.pTs,
        fi.floris.farm.turbine_fTilts,
        fi.floris.farm.correct_cp_ct_for_tilt,
        fi.floris.farm.turbine_type_map,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        tilt_angles,
        ref_tilt_cp_cts,
        fi.floris.farm.turbine_fCts,
        fi.floris.farm.turbine_fTilts,
        fi.floris.farm.correct_cp_ct_for_tilt,
        fi.floris.farm.turbine_type_map,
    )
    farm_powers = power(
        fi.floris.farm.ref_density_cp_cts,
        farm_eff_velocities,
        fi.floris.farm.turbine_power_interps,
        fi.floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        ref_tilt_cp_cts,
        fi.floris.farm.turbine_fCts,
        fi.floris.farm.turbine_fTilts,
        fi.floris.farm.correct_cp_ct_for_tilt,
        fi.floris.farm.turbine_type_map,
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
