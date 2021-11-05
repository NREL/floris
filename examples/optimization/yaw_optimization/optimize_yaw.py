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


def plot_hor_slice(fi):
    # Initialize the horizontal cut
    hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)

    # Plot and show
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    return fig, ax


if __name__ == "__main__":
    # Load FLORIS
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    print("Running FLORIS with no yaw misalignment...")
    fi.calculate_wake()
    power_initial = fi.get_farm_power()

    # =============================================================================
    print("Plotting the FLORIS flowfield...")
    # =============================================================================
    fig, ax = plot_hor_slice(fi)
    ax.set_title("Baseline Case for U = 8 m/s, Wind Direction = 270$^\\circ$")

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS...")
    # =============================================================================
    # Instantiate the Serial Optimization (SR) Optimization object. This optimizer
    # uses the Serial Refinement approach from Fleming et al. to quickly converge
    # close to the optimal solution in a minimum number of function evaluations.
    # Then, it will refine the optimal solution using the SciPy minimize() function.
    yaw_opt = YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),  # Yaw angles for baseline case
        bnds=[[-25.0, 25.0], [0.0, 20.0], [-3.0, 3.0]],  # Allowable yaw angles
        include_unc=False,  # No wind direction variability in floris simulations
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
        cluster_turbines=False,  # Do not bother with clustering
    )
    yaw_angles = yaw_opt.optimize()  # Perform optimization

    print("==========================================")
    print("yaw angles = ")
    for i in range(len(yaw_angles)):
        print("Turbine ", i, "=", yaw_angles[i], " deg")

    # Assign yaw angles to turbines and calculate wake
    fi.calculate_wake(yaw_angles=yaw_angles)
    power_opt = fi.get_farm_power()

    print("==========================================")
    print(
        "Total Power Gain = %.1f%%"
        % (100.0 * (power_opt - power_initial) / power_initial)
    )
    print("==========================================")
    # =============================================================================
    print("Plotting the FLORIS flowfield with yaw...")
    # =============================================================================
    fig, ax = plot_hor_slice(fi)
    ax.set_title("Optimal Wake Steering for U = 8 m/s, Wind Direction = 270$^\\circ$")
    plt.show()
