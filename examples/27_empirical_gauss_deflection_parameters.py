# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/floris/intro.html for documentation


import copy

import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface
from floris.tools.visualization import plot_rotor_values, visualize_cut_plane


"""
This example illustrates the main parameters of the Empirical Gaussian
deflection model and their effects on the wind turbine wake.
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.

# Options
show_flow_cuts = True
num_in_row = 5 # Should be at least 3
first_three_yaw_angles = [20., 20., 10.]

yaw_angles = np.array(first_three_yaw_angles + [0.]*(num_in_row-3))\
    [None, None, :]

print("Turbine yaw angles (degrees): ", yaw_angles[0,0,:])

# Define function for visualizing wakes
def generate_wake_visualization(fi, title=None):
    # Using the FlorisInterface functions, get 2D slices.
    x_bounds = [-500, 3000]
    y_bounds = [-250, 250]
    z_bounds = [0.001, 500]
    cross_plane_locations = [10, 1200, 2500]
    horizontal_plane_location = 90.0
    streamwise_plane_location = 0.0
    # Contour plot colors
    min_ws = 4
    max_ws = 10

    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=horizontal_plane_location,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        yaw_angles=yaw_angles
    )
    y_plane = fi.calculate_y_plane(
        x_resolution=200,
        z_resolution=100,
        crossstream_dist=streamwise_plane_location,
        x_bounds=x_bounds,
        z_bounds=z_bounds,
        yaw_angles=yaw_angles
    )
    cross_planes = []
    for cpl in cross_plane_locations:
        cross_planes.append(
            fi.calculate_cross_plane(
                y_resolution=100,
                z_resolution=100,
                downstream_dist=cpl
            )
        )

    # Create the plots
    # Cutplane settings
    cp_ls = "solid" # line style
    cp_lw = 0.5 # line width
    cp_clr = "black" # line color
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    # Horizontal profile
    ax = fig.add_subplot(311)
    visualize_cut_plane(horizontal_plane, ax=ax, title="Top-down profile",
        min_speed=min_ws, max_speed=max_ws)
    ax.plot(x_bounds, [streamwise_plane_location]*2, color=cp_clr,
        linewidth=cp_lw, linestyle=cp_ls)
    for cpl in cross_plane_locations:
        ax.plot([cpl]*2, y_bounds, color=cp_clr, linewidth=cp_lw,
            linestyle=cp_ls)

    ax = fig.add_subplot(312)
    visualize_cut_plane(y_plane, ax=ax, title="Streamwise profile",
        min_speed=min_ws, max_speed=max_ws)
    ax.plot(x_bounds, [horizontal_plane_location]*2, color=cp_clr,
        linewidth=cp_lw, linestyle=cp_ls)
    for cpl in cross_plane_locations:
        ax.plot([cpl, cpl], z_bounds, color=cp_clr, linewidth=cp_lw,
            linestyle=cp_ls)

    # Spanwise profiles
    for i, (cp, cpl) in enumerate(zip(cross_planes, cross_plane_locations)):
        visualize_cut_plane(cp, ax=fig.add_subplot(3, len(cross_planes), i+7),
            title="Loc: {:.0f}m".format(cpl), min_speed=min_ws,
            max_speed=max_ws)

    # Add overall figure title
    if title is not None:
        fig.suptitle(title, fontsize=16)


## Main script

# Load input yaml and define farm layout
fi = FlorisInterface("inputs/emgauss.yaml")
D = fi.floris.farm.rotor_diameters[0]
fi.reinitialize(
    layout_x=[x*5.0*D for x in range(num_in_row)],
    layout_y=[0.0]*num_in_row,
    wind_speeds=[8.0],
    wind_directions=[270.0]
)

# Save dictionary to modify later
fi_dict = fi.floris.as_dict()

# Run wake calculation
fi.calculate_wake(yaw_angles=yaw_angles)

# Look at the powers of each turbine
turbine_powers = fi.get_turbine_powers().flatten()/1e6

fig0, ax0 = plt.subplots(1,1)
width = 0.1
nw = -2
x = np.array(range(num_in_row))+width*nw
nw += 1

title = "Original"
ax0.bar(x, turbine_powers, width=width, label=title)
ax0.legend()

# Visualize wakes
if show_flow_cuts:
    generate_wake_visualization(fi, title)

# Increase the maximum deflection attained
fi_dict_mod = copy.deepcopy(fi_dict)

fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
    ['horizontal_deflection_gain_D'] = 5.0

fi = FlorisInterface(fi_dict_mod)
fi.reinitialize(
    wind_speeds=[8.0],
    wind_directions=[270.0]
)

fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers().flatten()/1e6

x = np.array(range(num_in_row))+width*nw
nw += 1

title = "Increase max deflection"
ax0.bar(x, turbine_powers, width=width, label=title)

if show_flow_cuts:
    generate_wake_visualization(fi, title)

# Add (increase) influence of wake added mixing
fi_dict_mod = copy.deepcopy(fi_dict)
fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
   ['mixing_gain_deflection'] = 100.0

fi = FlorisInterface(fi_dict_mod)
fi.reinitialize(
    wind_speeds=[8.0],
    wind_directions=[270.0]
)

fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers().flatten()/1e6

x = np.array(range(num_in_row))+width*nw
nw += 1

title = "Increase mixing gain"
ax0.bar(x, turbine_powers, width=width, label=title)

if show_flow_cuts:
    generate_wake_visualization(fi, title)

# Add (increase) the yaw-added mixing contribution
fi_dict_mod = copy.deepcopy(fi_dict)
# Include a WIM gain so that YAM is reflected in deflection as well
# as deficit
fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
   ['mixing_gain_deflection'] = 100.0
fi_dict_mod['wake']['wake_deflection_parameters']['empirical_gauss']\
   ['yaw_added_mixing_gain'] = 1.0
fi = FlorisInterface(fi_dict_mod)
fi.reinitialize(
    wind_speeds=[8.0],
    wind_directions=[270.0]
)

fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers().flatten()/1e6

x = np.array(range(num_in_row))+width*nw
nw += 1

title = "Increase yaw-added mixing"
ax0.bar(x, turbine_powers, width=width, label=title)

if show_flow_cuts:
    generate_wake_visualization(fi, title)

# Power plot aesthetics
ax0.set_xticks(range(num_in_row))
ax0.set_xticklabels(["T{0}".format(t) for t in range(num_in_row)])
ax0.legend()
ax0.set_xlabel("Turbine")
ax0.set_ylabel("Power [MW]")

plt.show()
