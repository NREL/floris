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

from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane

"""
This example initializes the FLORIS software, and then uses internal
functions to run a simulation and plot the results. In this case,
we are plotting three slices of the resulting flow field:
1. Horizontal slice parallel to the ground and located at the hub height
2. Vertical slice of parallel with the direction of the wind
3. Veritical slice parallel to to the turbine disc plane
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")

# Using the FlorisInterface functions for generating plots, run FLORIS
# and extract 2D planes of data.
horizontal_plane = fi.get_hor_plane(x_resolution=200, y_resolution=100)
y_plane = fi.get_y_plane(x_resolution=200, z_resolution=100)
cross_plane = fi.get_cross_plane(y_resolution=100, z_resolution=100)

# Create the plots
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
visualize_cut_plane(horizontal_plane, ax=ax_list[0], title="Horizontal")
visualize_cut_plane(y_plane, ax=ax_list[1], title="Streamwise profile")
visualize_cut_plane(cross_plane, ax=ax_list[2], title="Spanwise profile")

plt.show()
