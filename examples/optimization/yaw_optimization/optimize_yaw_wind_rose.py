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
from time import perf_counter as timerpc

import numpy as np

import floris.tools as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)
from floris.tools.optimization.yaw_optimization.yaw_optimizer_wind_rose_serial import (
    YawOptimizationWindRose,
)


def load_floris():
    # Instantiate the FLORIS object
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(file_dir, "../../example_input.json")
    )

    # Set turbine locations to 3 turbines in a row
    D = fi.floris.farm.turbines[0].rotor_diameter
    layout_x = [0, 7 * D, 14 * D, 0, 7 * D, 14 * D]
    layout_y = [0, 0, 0, 5 * D, 5 * D, 5 * D]
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    return fi


def load_optimizer():
    return YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),  # Yaw angles for baseline case
        minimum_yaw_angle=0.0,  # Allowable yaw angles lower bound
        maximum_yaw_angle=25.0,  # Allowable yaw angles upper bound
        include_unc=False,  # No wind direction variability in floris simulations
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
        cluster_turbines=False,  # Do not bother with clustering
    )


if __name__ == "__main__":
    # Load FLORIS
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # Load yaw optimizer and wind rose wrapper
    start_time = timerpc()
    wd_array = np.arange(0.0, 360.0, 30.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    ti_array = 0.06 * np.ones_like(wd_array)
    yaw_opt = YawOptimizationWindRose(
        yaw_optimization_obj=load_optimizer(),
        wd_array=wd_array,
        ws_array=ws_array,
        ti_array=ti_array,
    )

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS...")
    # =============================================================================
    # Instantiate the Serial Optimization (SR) Optimization object. This optimizer
    # uses the Serial Refinement approach from Fleming et al. to quickly converge
    # close to the optimal solution in a minimum number of function evaluations.
    # Then, it will refine the optimal solution using the SciPy minimize() function.
    df_opt = yaw_opt.optimize()
    end_time = timerpc()

    print("==========================================")
    print("Total Power Gain = ")
    for i in df_opt.index:
        wd = df_opt.loc[i, "wd"]
        Pbl = df_opt.loc[i, "power_baseline_weighted"]
        Popt = df_opt.loc[i, "power_opt_weighted"]
        print(
            "Case [%d]: wd = %.2f deg. Gain: %.3f %%"
            % (i, wd, 100.0 * (Popt - Pbl) / Pbl)
        )
    print("==========================================")

    print("==========================================")
    print("Total time spent: %.2f s" % (end_time - start_time))
    print("==========================================")
