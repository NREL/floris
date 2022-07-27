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
from floris.tools.visualization import visualize_cut_plane

"""
05_sweep_wind_speeds

This example demonstrates vectorization of wind speed.  
A vector of wind speeds is passed to the intialize function 
and the powers of the two simulated turbines is computed for all
wind speeds in one call

The power of both turbines for each wind speed is then plotted

"""


# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Define a two turbine farm
D = 126.
layout_x = np.array([0, D*6])
layout_y = [0, 0]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# Sweep wind speeds but keep wind direction fixed
ws_array = np.arange(5,25,0.5)
fi.reinitialize(wind_speeds=ws_array)

# Define a matrix of yaw angles to be all 0
# Note that yaw angles is now specified as a matrix whose dimesions are
# wd/ws/turbine
num_wd = 1
num_ws = len(ws_array)
num_turbine = len(layout_x)
yaw_angles = np.zeros((num_wd, num_ws, num_turbine)) 

# Calculate
fi.calculate_wake(yaw_angles=yaw_angles)

# Collect the turbine powers
turbine_powers = fi.get_turbine_powers() / 1E3 # In kW

# Pull out the power values per turbine
pow_t0 = turbine_powers[:,:,0].flatten()
pow_t1 = turbine_powers[:,:,1].flatten()

# Plot
fig, ax = plt.subplots()
ax.plot(ws_array,pow_t0,color='k',label='Upstream Turbine')
ax.plot(ws_array,pow_t1,color='r',label='Downstream Turbine')
ax.grid(True)
ax.legend()
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Power (kW)')
plt.show()
