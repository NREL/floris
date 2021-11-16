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


import os

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)


def load_floris():
    # Instantiate the FLORIS object
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(file_dir, "../../example_input.json")
    )

    # Set turbine locations to 3 turbines in a row
    D = fi.floris.farm.turbines[0].rotor_diameter
    layout_x = [0, 7 * D, 14 * D]
    layout_y = [0, 0, 0]
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    return fi


if __name__ == "__main__":
    # Load FLORIS
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS without uncertainty...")
    # =============================================================================
    # Instantiate the Serial Optimization (SR) Optimization object. This optimizer
    # uses the Serial Refinement approach from Fleming et al. to quickly converge
    # close to the optimal solution in a minimum number of function evaluations.
    # Then, it will refine the optimal solution using the SciPy minimize() function.
    yaw_opt = YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        include_unc=False,
    )
    yaw_angles_deterministic = yaw_opt.optimize()

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS with uncertainty...")
    # =============================================================================
    yaw_opt = YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        include_unc=True,
        unc_options={
            "std_wd": 5.0,
            "std_yaw": 0.0,
            "pmf_res": 1.0,
            "pdf_cutoff": 0.995,
        },
    )
    yaw_angles_stochastic = yaw_opt.optimize()

    print("==========================================")
    print("yaw angles (deterministic) = ", yaw_angles_deterministic)
    print("yaw angles (stochastic) = ", yaw_angles_stochastic)

    # Get baseline power productions
    fi.calculate_wake(yaw_angles=np.zeros(num_turbs))
    Pbl_wdstd0p00 = fi.get_farm_power(include_unc=False)
    Pbl_wdstd5p00 = fi.get_farm_power(
        include_unc=True,
        unc_options={"std_wd": 5, "std_yaw": 0, "pmf_res": 1, "pdf_cutoff": 0.995},
    )

    # Check performance baseline yaw angles for two different uncertainty levels
    fi.calculate_wake(yaw_angles=yaw_angles_deterministic)
    Popt_deterministic_wdstd0p00 = fi.get_farm_power(include_unc=False)
    Popt_deterministic_wdstd5p00 = fi.get_farm_power(
        include_unc=True,
        unc_options={"std_wd": 5, "std_yaw": 0, "pmf_res": 1, "pdf_cutoff": 0.995},
    )

    # Check performance robust yaw angles for two different uncertainty levels
    fi.calculate_wake(yaw_angles=yaw_angles_stochastic)
    Popt_stochastic_wdstd0p00 = fi.get_farm_power(include_unc=False)
    Popt_stochastic_wdstd5p00 = fi.get_farm_power(
        include_unc=True,
        unc_options={"std_wd": 5, "std_yaw": 0, "pmf_res": 1, "pdf_cutoff": 0.995},
    )

    print("==========================================")
    print(
        "Total Power Gain without Uncertainty = %.1f%%"
        % (100.0 * (Popt_deterministic_wdstd0p00 - Pbl_wdstd0p00) / Pbl_wdstd0p00)
    )
    print(
        "Total power gain with uncertainty using deterministic yaw angles = %.2f%%"
        % (100.0 * (Popt_deterministic_wdstd5p00 - Pbl_wdstd5p00) / Pbl_wdstd5p00)
    )
    print(
        "Total power gain with uncertainty using robust yaw angles = %.2f%%"
        % (100.0 * (Popt_stochastic_wdstd5p00 - Pbl_wdstd5p00) / Pbl_wdstd5p00)
    )
    print("==========================================")
