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

from floris.tools import FlorisInterface
from floris.tools import visualize_cut_plane #, plot_turbines_with_fi

"""
This example makes changes to the given input file through the script.
First, we plot simulation from the input file as given. Then, we make a series
of changes and generate plots from those simulations.
"""

# Create the plotting objects using matplotlib
fig, axarr = plt.subplots(2, 3, figsize=(12, 5))
axarr = axarr.flatten()

MIN_WS = 1.0
MAX_WS = 8.0

# Initialize FLORIS with the given input file via FlorisInterface
fi = FlorisInterface("inputs/gch.yaml")


# Plot a horizatonal slice of the initial configuration
horizontal_plane = fi.calculate_horizontal_plane(height=90.0)
visualize_cut_plane(horizontal_plane, ax=axarr[0], title="Initial setup", minSpeed=MIN_WS, maxSpeed=MAX_WS)


# Change the wind speed
horizontal_plane = fi.calculate_horizontal_plane(ws=[7.0], height=90.0)
visualize_cut_plane(horizontal_plane, ax=axarr[1], title="Wind speed at 7 m/s", minSpeed=MIN_WS, maxSpeed=MAX_WS)


# Change the wind shear, reset the wind speed, and plot a vertical slice
fi.reinitialize( wind_shear=0.2, wind_speeds=[8.0] )
y_plane = fi.calculate_y_plane(crossstream_dist=0.0)
visualize_cut_plane(y_plane, ax=axarr[2], title="Wind shear at 0.2", minSpeed=MIN_WS, maxSpeed=MAX_WS)


# Change the farm layout
N = 3  # Number of turbines per row and per column
X, Y = np.meshgrid(
    5.0 * fi.floris.farm.rotor_diameters[0][0][0] * np.arange(0, N, 1),
    5.0 * fi.floris.farm.rotor_diameters[0][0][0] * np.arange(0, N, 1),
)
fi.reinitialize( layout=( X.flatten(), Y.flatten() ) )
horizontal_plane = fi.calculate_horizontal_plane(height=90.0)
visualize_cut_plane(horizontal_plane, ax=axarr[3], title="3x3 Farm", minSpeed=MIN_WS, maxSpeed=MAX_WS)


# Change the yaw angles and configure the plot differently
yaw_angles = np.zeros((1, 1, N * N))

## First row
yaw_angles[:,:,0] = 30.0
yaw_angles[:,:,1] = -30.0
yaw_angles[:,:,2] = 30.0

## Second row
yaw_angles[:,:,3] = -30.0
yaw_angles[:,:,4] = 30.0
yaw_angles[:,:,5] = -30.0

horizontal_plane = fi.calculate_horizontal_plane(yaw_angles=yaw_angles, height=90.0)
visualize_cut_plane(horizontal_plane, ax=axarr[4], title="Yawesome art", cmap="PuOr", minSpeed=MIN_WS, maxSpeed=MAX_WS)
# plot_turbines_with_fi(axarr[8], fi)


# Plot the cross-plane of the 3x3 configuration
cross_plane = fi.calculate_cross_plane(yaw_angles=yaw_angles, downstream_dist=610.0)
visualize_cut_plane(cross_plane, ax=axarr[5], title="Cross section at 610 m", minSpeed=MIN_WS, maxSpeed=MAX_WS)
axarr[5].invert_xaxis()


plt.show()