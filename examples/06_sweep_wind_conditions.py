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


import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


"""
This example demonstrates the vectorized wake calculation for
a set of wind speeds and directions combinations. When given
a list of conditions, FLORIS leverages features of the CPU
to perform chunks of the computations at once rather than
looping over each condition.

This calculation is performed for a single-row 5 turbine farm.  In addition
to plotting the powers of the individual turbines, an energy by turbine
calculation is made and plotted by summing over the wind speed and wind direction
axes of the power matrix returned by get_turbine_powers()

"""

# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml")  # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Define a 5 turbine farm
D = 126.0
layout_x = np.array([0, D*6, D*12, D*18, D*24])
layout_y = [0, 0, 0, 0, 0]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# In this case we want to check a grid of wind speed and direction combinations
wind_speeds_to_expand = np.arange(6, 9, 1.0)
wind_directions_to_expand = np.arange(250, 295, 1.0)
num_unique_ws = len(wind_speeds_to_expand)
num_unique_wd = len(wind_directions_to_expand)

# Create grids to make combinations of ws/wd
wind_speeds_grid, wind_directions_grid = np.meshgrid(
    wind_speeds_to_expand,
    wind_directions_to_expand
)

# Flatten the grids back to 1D arrays
ws_array = wind_speeds_grid.flatten()
wd_array = wind_directions_grid.flatten()

# Now reinitialize FLORIS
fi.reinitialize(wind_speeds=ws_array, wind_directions=wd_array)

# Define a matrix of yaw angles to be all 0
# Note that yaw angles is now specified as a matrix whose dimensions are
# (findex, turbine)
num_wd = len(wd_array)
num_ws = len(ws_array)
n_findex = num_wd  # Could be either num_wd or num_ws
num_turbine = len(layout_x)
yaw_angles = np.zeros((n_findex, num_turbine))

# Calculate
fi.calculate_wake(yaw_angles=yaw_angles)

# Collect the turbine powers
turbine_powers = fi.get_turbine_powers() / 1e3  # In kW

# Show results by ws and wd
fig, axarr = plt.subplots(num_unique_ws, 1, sharex=True, sharey=True, figsize=(6, 10))
for ws_idx, ws in enumerate(wind_speeds_to_expand):
    indices = ws_array == ws
    ax = axarr[ws_idx]
    for t in range(num_turbine):
        ax.plot(wd_array[indices], turbine_powers[indices, t].flatten(), label="T%d" % t)
    ax.legend()
    ax.grid(True)
    ax.set_title("Wind Speed = %.1f" % ws)
    ax.set_ylabel("Power (kW)")
ax.set_xlabel("Wind Direction (deg)")

# Sum across wind speeds and directions to show energy produced by turbine as bar plot
# Sum over wind directions and speeds
energy_by_turbine = np.sum(turbine_powers, axis=0)
fig, ax = plt.subplots()
ax.bar(["T%d" % t for t in range(num_turbine)], energy_by_turbine)
ax.set_title("Energy Produced by Turbine")

plt.show()
