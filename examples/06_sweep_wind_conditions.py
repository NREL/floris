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
06_sweep_wind_conditions

This example demonstrates vectorization of wind speed and wind direction.
When the intialize function is passed an array of wind speeds and an
array of wind directions it automatically expands the vectors to compute
the result of all combinations.

This calculation is performed for a single-row 5 turbine farm.  In addition
to plotting the powers of the individual turbines, an energy by turbine
calculation is made and plotted by summing over the wind speed and wind direction
axes of the power matrix returned by get_turbine_powers()

"""

# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Define a 5 turbine farm
D = 126.
layout_x = np.array([0, D*6, D*12, D*18,D*24])
layout_y = [0, 0, 0, 0, 0]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# Define a ws and wd to sweep
# Note that all combinations will be computed
ws_array = np.arange(6, 9, 1.)
wd_array = np.arange(250,295,1.)
fi.reinitialize(wind_speeds=ws_array, wind_directions=wd_array)

# Define a matrix of yaw angles to be all 0
# Note that yaw angles is now specified as a matrix whose dimesions are
# wd/ws/turbine
num_wd = len(wd_array)
num_ws = len(ws_array)
num_turbine = len(layout_x)
yaw_angles = np.zeros((num_wd, num_ws, num_turbine))

# Calculate
fi.calculate_wake(yaw_angles=yaw_angles)

# Collect the turbine powers
turbine_powers = fi.get_turbine_powers() / 1E3 # In kW

# Show results by ws and wd
fig, axarr = plt.subplots(num_ws, 1, sharex=True,sharey=True,figsize=(6,10))
for ws_idx, ws in enumerate(ws_array):
    ax = axarr[ws_idx]
    for t in range(num_turbine):
        ax.plot(wd_array, turbine_powers[:,ws_idx,t].flatten(),label='T%d' % t)
    ax.legend()
    ax.grid(True)
    ax.set_title('Wind Speed = %.1f' % ws)
    ax.set_ylabel('Power (kW)')
ax.set_xlabel('Wind Direction (deg)')

# Sum across wind speeds and directions to show energy produced by turbine as bar plot
# Sum over wind direction (0-axis) and wind speed (1-axis)
energy_by_turbine = np.sum(turbine_powers, axis=(0,1))
fig, ax = plt.subplots()
ax.bar(['T%d' % t for t in range(num_turbine)],energy_by_turbine)
ax.set_title('Energy Produced by Turbine')

plt.show()
