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
    power,
    thrust_coefficient,
)
from floris.simulation.rotor_velocity import rotor_effective_velocity
from floris.tools import FlorisInterface
from tests.conftest import (
    assert_results_arrays,
    N_FINDEX,
    N_TURBINES,
    print_test_values,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
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
    n_findex = fi.floris.flow_field.n_findex

    velocities = fi.floris.flow_field.u
    yaw_angles = fi.floris.farm.yaw_angles
    tilt_angles = fi.floris.farm.tilt_angles
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        yaw_angles,
        tilt_angles,
        fi.floris.farm.turbine_thrust_coefficient_functions,
        fi.floris.farm.turbine_tilt_interps,
        fi.floris.farm.correct_cp_ct_for_tilt,
        fi.floris.farm.turbine_type_map,
        fi.floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        fi.floris.flow_field.air_density,
        fi.floris.farm.turbine_power_functions,
        fi.floris.farm.yaw_angles,
        fi.floris.farm.tilt_angles,
        fi.floris.farm.turbine_tilt_interps,
        fi.floris.farm.turbine_type_map,
        fi.floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        tilt_angles,
        fi.floris.farm.turbine_axial_induction_functions,
        fi.floris.farm.turbine_tilt_interps,
        fi.floris.farm.correct_cp_ct_for_tilt,
        fi.floris.farm.turbine_type_map,
        fi.floris.farm.turbine_power_thrust_tables,
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
