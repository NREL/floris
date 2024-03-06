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
import yaml

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface


"""
This example initializes the FLORIS software, and then uses internal
functions to run a simulation and plot the results. In this case,
we are plotting three slices of the resulting flow field:
1. Horizontal slice parallel to the ground and located at the hub height
2. Vertical slice of parallel with the direction of the wind
3. Veritical slice parallel to to the turbine disc plane

Additionally, an alternative method of plotting a horizontal slice
is shown. Rather than calculating points in the domain behind a turbine,
this method adds an additional turbine to the farm and moves it to
locations throughout the farm while calculating the velocity at it's
rotor.
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/emgauss.yaml")

with open(str(
    fi.floris.as_dict()["farm"]["turbine_library_path"] /
    (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "helix"

fi.reinitialize(layout_x=[0, 0.0, 1000, 1000], layout_y=[0.0, 600.0, 0, 600], turbine_type=['iea_15mw'])

# Set the wind directions and speeds to be constant over n_findex = N time steps
N = 1
fi.reinitialize(wind_directions=270 * np.ones(N), wind_speeds=10.0 * np.ones(N))
fi.calculate_wake()
turbine_powers_orig = fi.get_turbine_powers()

# Add helix
helix_amplitudes = np.array([5, 0, 0, 0]).reshape(4, N).T
fi.calculate_wake(helix_amplitudes=helix_amplitudes)
turbine_powers_helix = fi.get_turbine_powers()

# The rotor plots show what is happening at each turbine, but we do not
# see what is happening between each turbine. For this, we use a
# grid that has points regularly distributed throughout the fluid domain.
# The FlorisInterface contains functions for configuring the new grid,
# running the simulation, and generating plots of 2D slices of the
# flow field.

# Note this visualization grid created within the calculate_horizontal_plane function will be reset
# to what existed previously at the end of the function

# Using the FlorisInterface functions, get 2D slices.
horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=150.0,
    yaw_angles=np.array([[0.,0.,0,0]]),
)

y_plane = fi.calculate_y_plane(
    x_resolution=200,
    z_resolution=100,
    crossstream_dist=0.0,
    yaw_angles=np.array([[0.,0.,0,0]]),
)
cross_plane = fi.calculate_cross_plane(
    y_resolution=100,
    z_resolution=100,
    downstream_dist=1200.0,
    yaw_angles=np.array([[0.,0.,0,0]]),
)

# Create the plots
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
wakeviz.visualize_cut_plane(
    horizontal_plane,
    ax=ax_list[0],
    label_contours=True,
    title="Horizontal"
)
wakeviz.visualize_cut_plane(
    y_plane,
    ax=ax_list[1],
    label_contours=True,
    title="Streamwise profile"
)
wakeviz.visualize_cut_plane(
    cross_plane,
    ax=ax_list[2],
    label_contours=True,
    title="Spanwise profile"
)

wakeviz.show_plots()
