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

# Set the wind direction to run 360 degrees
wd_array = np.arange(0, 360, 1)
fi.reinitialize(wind_directions=wd_array)

# Grab the power before we change anything
fi.calculate_wake()

# Simulate a met mast in between the turbines
points_x = [3*D, 3*D, 3*D, 3*D]
points_y = [0, 0, 0, 0]
points_z = [30, 90, 150, 250]

# Collect the points
u_at_points = fi.sample_flow_at_points(points_x = points_x,
                                        points_y = points_y,
                                          points_z = points_z)


# Plot the velocities
fig, ax = plt.subplots()
for z_idx, z in enumerate(points_z):
    ax.plot(wd_array, u_at_points[:, z_idx].flatten(), label=f'Speed at {z} m')
ax.grid()
ax.legend()
ax.set_xlabel('Wind Direction (deg)')
ax.set_ylabel('Wind Speed (m/s)')

plt.show()
