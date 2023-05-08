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


import matplotlib.pyplot as plt
import numpy as np

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
from floris.tools.visualization import (
    calculate_horizontal_plane_with_turbines,
    visualize_cut_plane,
)


"""
Test the use of add points
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")


# Set up a two-turbine farm
D = 126
fi.reinitialize(layout_x=[0, 3*D], layout_y=[0, 3*D])

fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,4)
ax[0].scatter(fi.layout_x, fi.layout_y, color="black", label="turbine")

# Set the wind direction to run 360 degrees
wd_array = np.arange(0, 360, 1)
fi.reinitialize(wind_directions=wd_array)

# Grab the power before we change anything
fi.calculate_wake()

# Simulate a met mast in between the turbines
met_mast_option = 0 # Try 0, 1, 2, 3 (requires python >= 3.10)
match met_mast_option:
    case 0:
        points_x = [3*D]*4
        points_y = [0]*4
    case 1:
        points_x = [200.]*4
        points_y = [200.]*4
    case 2:
        points_x = [20.]*4
        points_y = [20.]*4
    case 3:
        points_x = [305.]*4
        points_y = [158.]*4

points_z = [30, 90, 150, 250]

ax[0].scatter(points_x, points_y, color="red", marker="x", label="test point")
ax[0].grid()
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
ax[0].legend()

# Collect the points
u_at_points = fi.sample_flow_at_points(points_x = points_x,
                                        points_y = points_y,
                                          points_z = points_z)


# Plot the velocities
for z_idx, z in enumerate(points_z):
    ax[1].plot(wd_array, u_at_points[:, :, z_idx].flatten(), label=f'Speed at {z} m')
ax[1].grid()
ax[1].legend()
ax[1].set_xlabel('Wind Direction (deg)')
ax[1].set_ylabel('Wind Speed (m/s)')

plt.show()