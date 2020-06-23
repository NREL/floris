# Copyright 2020 NREL

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

import pyoptsparse
import floris.tools as wfct


# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, "../../../example_input.json")
)

# Set turbine locations to 4 turbines in a rectangle
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 0, 6 * D, 6 * D]
layout_y = [0, 5 * D, 0, 5 * D]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

nturbs = len(fi.floris.farm.turbines)

# Generate random wind rose data
wd = np.arange(0.0, 360.0, 90.0)
np.random.seed(1)
ws = 8.0 + np.random.randn(len(wd)) * 0.5
freq = np.abs(np.sort(np.random.randn(len(wd))))
freq = freq / freq.sum()


def objective_function(varDict):
    # Parse the variable dictionary
    yaw = varDict["yaw"]
    AEP_sum = 0

    fi.reinitialize_flow_field(wind_direction=[wd_itr], wind_speed=[ws_itr])

    AEP_sum = AEP_sum - fi.get_farm_power_for_yaw_angle(yaw) * freq_itr * 8760

    # Compute the objective function
    funcs = {}
    funcs["obj"] = AEP_sum / 1e9

    fail = False
    return funcs, fail


solutions = []

for i in range(len(wd)):
    wd_itr = wd[i]
    ws_itr = ws[i]
    freq_itr = freq[i]

    # Setup the optimization problem
    optProb = pyoptsparse.Optimization("yaw_opt_wd_" + str(wd_itr), objective_function)

    # Add the design variables to the optimization problem
    optProb.addVarGroup("yaw", nturbs, "c", lower=0.0, upper=20.0, value=2.0)

    # Add the objective to the optimization problem
    optProb.addObj("obj")

    # Setup the optimization solver
    # Note: pyOptSparse has other solvers available; some may require additional
    #   licenses/installation. See https://github.com/mdolab/pyoptsparse for
    #   more information. When ready, they can be invoked by changing 'SLSQP'
    #   to the solver name, for example: 'opt = pyoptsparse.SNOPT(fi=fi)'.
    opt = pyoptsparse.SLSQP()

    # Run the optimization with finite-differencing
    solution = opt(optProb, sens="FDR")
    solutions.append(solution)

[print(sol) for sol in solutions]

print("\n" + "=".join(["="] * 41))
print("{:^80}".format("Summary of Yaw Optimization Results"))
print("=".join(["="] * 41) + "\n")
print(
    "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "wd", "ws", "T1 yaw", "T2 yaw", "T3 yaw", "T4 yaw"
    )
)
# print('-'.join(['-'])*81)

for i in range(len(wd)):
    print(
        "{:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f}".format(
            wd[i],
            ws[i],
            solutions[i].getDVs()["yaw"][0],
            solutions[i].getDVs()["yaw"][1],
            solutions[i].getDVs()["yaw"][2],
            solutions[i].getDVs()["yaw"][3],
        )
    )

print("\n" + "=".join(["="] * 41) + "\n")
