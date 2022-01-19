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


import os
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

import floris.tools as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)


def load_floris(N=2):
    # Load the default example floris object
    root_path = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(root_path, "..", "..", "example_input.yaml")
    )

    # Specify wind farm layout and update in the floris object
    X, Y = np.meshgrid(
        5.0 * fi.floris.grid.reference_turbine_diameter * np.arange(0, N, 1),
        5.0 * fi.floris.grid.reference_turbine_diameter * np.arange(0, N, 1),
    )
    fi.reinitialize(
        layout=(X.flatten(), Y.flatten()),
        wind_directions=np.arange(0.0, 360.0, 5.0),
        wind_speeds=[8.0],
    )

    return fi


if __name__ == "__main__":
    # Specify current directory
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Specify sqrt of number of turbines to iterate over
    N_array = [3, 5, 7, 9, 10]

    # Initialize empty matrices
    timings = np.zeros(len(N_array), dtype=float)
    gains = np.zeros(len(N_array), dtype=float)

    # Optimize for every wind farm size
    for ii, N in enumerate(N_array):
        # Load FLORIS for N^2 number of turbines
        fi = load_floris(N=N)

        # Now optimize the yaw angles using the Serial Refine method
        print("==============================================")
        print("Processing yaw optimization with N={:d}.".format(N))
        start_time = timerpc()
        yaw_opt = YawOptimizationSR(
            fi=fi,
            minimum_yaw_angle=0.0,  # Allowable yaw angles lower bound
            maximum_yaw_angle=20.0,  # Allowable yaw angles upper bound
            yaw_angles_baseline=np.zeros((72, 1, len(fi.layout_x))),
            Ny_passes=[5, 4],
            exploit_layout_symmetry=True,
            exclude_downstream_turbines=True,
        )
        df_opt = yaw_opt._optimize()
        t = timerpc() - start_time
        timings[ii] = t
        gains[ii] = 100.0 * (
            df_opt["farm_power_opt"].sum() / 
            df_opt["farm_power_baseline"].sum() - 1.0
        )
        print("Optimization finished in {:.2f} seconds.".format(t))
        fout = os.path.join(root_path, "df_opt_N{:d}_v30_serialrefine.csv".format(N))
        print("Saving optimization results to {:s}".format(fout))
        print(" ")

        # Save detailed optimal solutions to a .csv
        df_opt.to_csv(fout)

    # Save benchmarking results to a .csv
    df_benchmarking = pd.DataFrame(
        {"N": N_array, "timings": timings, "gains": gains}
    )
    root_path = os.path.dirname(os.path.abspath(__file__))
    fout = os.path.join(root_path, "benchmarking_results_v30_serialrefine.csv")
    df_benchmarking.to_csv(fout, index=False, mode="a")  # Append
    print("Benchmarking results saved to {:s}.".format(fout))
