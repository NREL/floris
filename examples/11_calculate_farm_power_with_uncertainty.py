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
04_sweep_wind_directions

This example demonstrates vectorization of wind direction.  
A vector of wind directions is passed to the intialize function 
and the powers of the two simulated turbines is computed for all
wind directions in one call

The power of both turbines for each wind direction is then plotted

"""

# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Define a two turbine farm
D = 126.
layout_x = np.array([0, D*6, D*12])
layout_y = [0, 0, 0]
fi.reinitialize(layout=[layout_x, layout_y])

# Sweep wind speeds but keep wind direction fixed
wd_array = np.arange(0.0, 360.0, 1.0)
fi.reinitialize(wind_directions=wd_array)

# Define a matrix of yaw angles to be all 0
# Note that yaw angles is now specified as a matrix whose dimesions are
# wd/ws/turbine
num_wd = len(wd_array) # Number of wind directions
num_ws = 1 # Number of wind speeds
num_turbine = len(layout_x) #  Number of turbines
yaw_angles = np.zeros((num_wd, num_ws, num_turbine)) 

# Calculate the nominal wake solution
fi.calculate_wake(yaw_angles=yaw_angles)

# Calculate the nominal wind farm power production
farm_powers_nom = fi.get_farm_power(include_unc=False) / 1e3

# Calculate the wind farm power with uncertainty on the wind direction
unc_options = {
    "std_wd": 3.0,  # Standard deviation for inflow wind direction (deg)
    "std_yaw": 0.0,  # Standard deviation in turbine yaw angle (deg)
    "pmf_res": 1.0,  # Resolution over which to calculate angles (deg)
    "pdf_cutoff": 0.995,  # Probability density function cut-off (-)
}
farm_powers_unc = fi.get_farm_power(
    include_unc=True,
    unc_options=unc_options
) / 1e3

# Plot results
fig, ax = plt.subplots()
ax.plot(wd_array, farm_powers_nom.flatten(), color='k',label='Nominal farm power')
ax.plot(wd_array, farm_powers_unc.flatten(), color='r',label='Farm power with uncertainty')
ax.grid(True)
ax.legend()
ax.set_xlabel('Wind Direction (deg)')
ax.set_ylabel('Power (kW)')
plt.show()
