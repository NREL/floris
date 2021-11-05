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
import matplotlib.pyplot as plt

import floris.tools as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import (
    YawOptimizationScipy,
)


def load_floris():
    # Instantiate the FLORIS object
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(file_dir, "../../example_input.json")
    )

    # Set turbine locations to 3 turbines in a row
    D = fi.floris.farm.turbines[0].rotor_diameter
    layout_x = np.array([0, 5, 10, 15, 0, 5, 10, 15, 0, 5, 10, 15]) * D
    layout_y = np.array([0, 0, 0, 0, 5, 5, 5, 5, 10, 10, 10, 10]) * D
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    return fi


def load_optimizer_slsqp(fi):
    return YawOptimizationScipy(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),  # Yaw angles for baseline case
        minimum_yaw_angle=0.0,  # Allowable yaw angles lower bound
        maximum_yaw_angle=25.0,  # Allowable yaw angles upper bound
        opt_options={
            "maxiter": 100,
            "disp": True,
            "iprint": 2,
            "ftol": 1e-12,
            "eps": 0.1,
        },
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
    )


def load_optimizer_sr(fi):
    return YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=np.zeros(num_turbs),  # Yaw angles for baseline case
        minimum_yaw_angle=0.0,  # Allowable yaw angles lower bound
        maximum_yaw_angle=25.0,  # Allowable yaw angles upper bound
        opt_options={
            "Ny_passes": [5, 5],
            "refine_solution": True,
            "refine_method": "SLSQP",
            "refine_options": {
                "maxiter": 10,
                "disp": True,
                "iprint": 2,
                "ftol": 1e-7,
                "eps": 0.01,
            },
        },
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
    )


if __name__ == "__main__":
    # Load FLORIS
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # Specify atmospheric conditions to optimize for
    wd_array = np.arange(0.0, 360.0, 30.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    ti_array = 0.06 * np.ones_like(wd_array)

    # Now optimize all atmospheric conditions in wd_array, ws_array and ti_array
    # using both serial refinement (SR) and SciPy's internal SLSQP optimization
    # code. The optimization will be timed from object initialization to executing
    # all optimizations successfully. The power gain/optimization performance is
    # also tested.
    dict_out = dict()
    for method in ["sr", "slsqp"]:
        print("==========================================")
        print("Calculating optimal solutions using %s." % method.upper())

        # Load the right yaw optimizer object
        if method == "sr":
            yaw_opt = load_optimizer_sr(fi=fi)
        else:
            yaw_opt = load_optimizer_slsqp(fi=fi)

        # Initialize empty variables
        clock_time = np.zeros_like(wd_array)
        gain_array = np.zeros_like(wd_array)

        # Calculate optimal solution for every set of inflow conditions
        for i in range(len(wd_array)):
            print("Optimizing case %d out of %d..." % (i, len(wd_array) - 1))
            # Time and optimize
            start_time = timerpc()
            yaw_opt.reinitialize_flow_field(
                wind_direction=wd_array[i],
                wind_speed=ws_array[i],
                turbulence_intensity=ti_array[i],
            )
            yaw_angles_opt = yaw_opt.optimize()
            end_time = timerpc()

            # Evaluate and get baseline and calculate optimized power
            yaw_opt.fi.reinitialize_flow_field(
                wind_direction=wd_array[i],
                wind_speed=ws_array[i],
                turbulence_intensity=ti_array[i],
            )

            yaw_opt.fi.calculate_wake(np.zeros(num_turbs))
            Pbl = yaw_opt.fi.get_farm_power()
            yaw_opt.fi.calculate_wake(yaw_angles_opt)
            Popt = yaw_opt.fi.get_farm_power()
            gain_array[i] = 100.0 * (Popt - Pbl) / Pbl
            clock_time[i] = end_time - start_time

        dict_out[method] = {"gain": gain_array, "time": clock_time}

    # Now print results
    print("==========================================")
    print("Total Power Gain = ")
    for i in range(len(wd_array)):
        wd = wd_array[i]
        gain = dict_out["sr"]["gain"][i]
        print("Case [%d]: wd = %.2f deg. Gain: %.3f %% [SR]" % (i, wd, gain))

        gain = dict_out["slsqp"]["gain"][i]
        print("Case [%d]: wd = %.2f deg. Gain: %.3f %% [SLSQP]" % (i, wd, gain))
        print(" ")
    print("==========================================")

    print("==========================================")
    print("Total time spent: %.2f s [SR]" % np.sum(dict_out["sr"]["time"]))
    print("Total time spent: %.2f s [SLSQP]" % np.sum(dict_out["slsqp"]["time"]))
    print("==========================================")

    # And make plots
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].bar(x=wd_array - 5, height=dict_out["sr"]["gain"], width=10, label="SR")
    ax[0].bar(x=wd_array + 5, height=dict_out["slsqp"]["gain"], width=10, label="SLSQP")
    ax[0].set_xticks(wd_array)
    ax[0].set_ylabel("Power gain (%)")
    ax[0].set_xlabel("Wind direction (deg)")
    ax[0].legend()

    ax[1].bar(x=wd_array - 5, height=dict_out["sr"]["time"], width=10, label="SR")
    ax[1].bar(x=wd_array + 5, height=dict_out["slsqp"]["time"], width=10, label="SLSQP")
    ax[1].set_xticks(wd_array)
    ax[1].set_ylabel("Computation time (s)")
    ax[1].set_xlabel("Wind direction (deg)")
    ax[1].legend()

    plt.show()
