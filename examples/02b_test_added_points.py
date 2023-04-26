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
fi.reinitialize(layout_x=[0, 7*D], layout_y=[0, 0])

# Grab the power before we change anything
fi.calculate_wake()
turbine_powers_before_added_points = np.array(fi.get_turbine_powers())/1000.

# Set up test points to run through the turbines
points_x = np.arange(D*-1, D*30, D/4)
points_y = np.zeros_like(points_x)
points_z = 90 * np.ones_like(points_x)

# Collect the points
u_at_points = fi.sample_flow_at_points(points_x = points_x,
                                       points_y = points_y,
                                       points_z = points_z)

# Re-collect the turbine powers
turbine_powers_after_added_points = np.array(fi.get_turbine_powers())/1000.

# Reinitialize and collect the turbine powers a third time
fi.calculate_wake()
turbine_powers_after_calc_again_points = np.array(fi.get_turbine_powers())/1000.

print('Turbine power before points: ', turbine_powers_before_added_points)
print('Turbine power after points: ', turbine_powers_after_added_points)
print('Turbine power after points and calc again: ', turbine_powers_after_calc_again_points)

# Get the same points using the turbine-based method
cut_plane_from_turbine_sweep = calculate_horizontal_plane_with_turbines(fi,
                                         x_resolution=100,
                                         y_resolution=1,
                                         x_bounds=[-D, 30*D],
                                         y_bounds=[0, 0],)

# Get the same points using horizontal plane
horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=1,
    x_bounds=[-D, 30*D],
    y_bounds=[0, 0],
    height=90.0,
    yaw_angles=np.array([[[0.,0.,0.]]]),
)


# Plot the points
fig, ax = plt.subplots()
ax.plot(cut_plane_from_turbine_sweep.df.x1/D,
        cut_plane_from_turbine_sweep.df.u,
          label='Velocity using turbine method',
            color='k')
ax.plot(horizontal_plane.df.x1/D,
         horizontal_plane.df.u,
           label='Velocity using horizontal plane',
           ls= '--')
ax.plot(points_x/D,
        u_at_points,
        label='Velocity at points',
          ls= '--',
          color='r')
ax.set_xlabel('X/D')
ax.set_ylabel('U (m/s)')
ax.grid()
ax.legend()

plt.show()
