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

import floris.tools.layout_functions as lf
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane


"""
This example shows a number of different ways to visualize a farm layout using FLORIS
"""

# Create the plotting objects using matplotlib
fig, axarr = plt.subplots(2, 3, figsize=(16, 10), sharex=False)
axarr = axarr.flatten()

MIN_WS = 1.0
MAX_WS = 8.0

# Initialize FLORIS with the given input file via FlorisInterface
fi = FlorisInterface("inputs/gch.yaml")

# Change to 2 x 3 layout with a wind direction from northwest
fi.reinitialize(
    layout_x=[0, 0, 1000, 1000, 1000], layout_y=[0, 500, 0, 500, 1000], wind_directions=[300]
)

# Plot 1: Visualize the flow
ax = axarr[0]
# Plot a horizatonal slice of the initial configuration
horizontal_plane = fi.calculate_horizontal_plane(height=90.0)
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    min_speed=MIN_WS,
    max_speed=MAX_WS,
)
# Plot the turbine points, setting the color to white
lf.plot_turbine_points(fi, ax=ax, plotting_dict={"color": "w"})
ax.set_title('Flow visualization and turbine points')

# Plot 2: Show a particular flow case
ax = axarr[1]
turbine_names = [f"T{i}" for i in [10, 11, 12, 13, 22]]
lf.plot_turbine_points(fi, ax=ax)
lf.plot_turbine_labels(fi, ax=ax, turbine_names=turbine_names, show_bbox=True, bbox_dict={'facecolor':'r'})
ax.set_title("Show turbine names with a red bounding box")


# Plot 2: Show turbine rotors on flow
ax = axarr[2]
horizontal_plane = fi.calculate_horizontal_plane(height=90.0, yaw_angles=np.array([[0., 30., 0., 0., 0.]]))
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    min_speed=MIN_WS,
    max_speed=MAX_WS
)
lf.plot_turbines_rotors(fi,ax=ax,yaw_angles=np.array([[0., 30., 0., 0., 0.]]))
ax.set_title("Flow visualization with yawed turbine")

# Plot 3: Show the layout, including wake directions
ax = axarr[3]
lf.plot_turbine_points(fi, ax=ax)
lf.plot_turbine_labels(fi, ax=ax, turbine_names=turbine_names)
lf.plot_waking_directions(fi, ax=ax)
ax.set_title("Show turbine names and wake direction")

# Plot 4: Plot a subset of the layout, and limit directions less than 7D
ax = axarr[4]
lf.plot_turbine_points(fi, ax=ax, turbine_indices=[0,1,2,3])
lf.plot_turbine_labels(fi, ax=ax, turbine_names=turbine_names, turbine_indices=[0,1,2,3])
lf.plot_waking_directions(fi, ax=ax, turbine_indices=[0,1,2,3], limit_dist_D=7)
ax.set_title("Plot a subset and limit wake line distance")

# Plot with a shaded region
ax = axarr[5]
lf.plot_turbine_points(fi, ax=ax)
lf.shade_region(np.array([[0,0],[300,0],[300,1000],[0,700]]),ax=ax)
ax.set_title("Plot with a shaded region")


plt.show()
