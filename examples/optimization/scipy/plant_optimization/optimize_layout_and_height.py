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
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.cut_plane as cp
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.layout_height import LayoutHeightOptimization


# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, "../../../example_input.json")
)

# Set turbine locations to 3 turbines in a triangle
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [10, 10, 10 + 7 * D]
layout_y = [200, 1000, 200]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Define the boundary for the wind farm
boundaries = [[2000.1, 4000.0], [2000.0, 0.1], [0.0, 0.0], [0.1, 2000.0]]

# Define the limits for the turbine height
height_lims = [85.0, 115.0]

# Definte the plant power rating in kW
plant_kw = 3 * 5000

# Generate random wind rose data
wd = np.arange(0.0, 360.0, 5.0)
np.random.seed(1)
ws = 8.0 + np.random.randn(len(wd)) * 0.5
freq = np.abs(np.sort(np.random.randn(len(wd))))
freq = freq / freq.sum()

# Set optimization options
opt_options = {"maxiter": 50, "disp": True, "iprint": 2, "ftol": 1e-8}

AEP_initial = fi.get_farm_AEP(wd, ws, freq)

# Instantiate the layout otpimization object
layout_height_opt = LayoutHeightOptimization(
    fi=fi,
    boundaries=boundaries,
    height_lims=height_lims,
    wd=wd,
    ws=ws,
    freq=freq,
    AEP_initial=AEP_initial,
    COE_initial=None,
    plant_kw=plant_kw,
    opt_options=opt_options,
)

# Compute initial COE for optimization normalization
COE_initial = layout_height_opt.COE_model.COE(
    height=fi.floris.farm.turbines[0].hub_height, AEP_sum=AEP_initial
)

print("COE_initial: ", COE_initial)

layout_height_opt.reinitialize_opt_height(COE_initial=COE_initial)

# Perform layout optimization
opt_results = layout_height_opt.optimize()

layout_results = opt_results[0]
height_results = opt_results[1]

print("=====================================================")
print("Layout coordinates: ")
for i in range(len(layout_results[0])):
    print(
        "Turbine",
        i,
        ": \tx = ",
        "{:.1f}".format(layout_results[0][i]),
        "\ty = ",
        "{:.1f}".format(layout_results[1][i]),
    )
print("Height: ", "{:.2f}".format(height_results[0]))

# Calculate new COE results
fi.reinitialize_flow_field(layout_array=(layout_results[0], layout_results[1]))
AEP_optimized = fi.get_farm_AEP(wd, ws, freq)
COE_optimized = layout_height_opt.COE_model.COE(
    height=height_results[0], AEP_sum=AEP_optimized
)

print("=====================================================")
print("COE Reduction = %.1f%%" % (100.0 * (COE_optimized - COE_initial) / COE_initial))
print("=====================================================")

# Plot the new layout vs the old layout
layout_height_opt.plot_layout_opt_results()
plt.show()
