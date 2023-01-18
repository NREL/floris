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

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface


"""
This example uses an input file where multiple turbine types are defined.
The first two turbines are the NREL 5MW, and the third turbine is the IEA 10MW.
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch_multiple_turbine_types.yaml")

# Using the FlorisInterface functions for generating plots, run FLORIS
# and extract 2D planes of data.
horizontal_plane = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=100, height=90)
y_plane = fi.calculate_y_plane(x_resolution=200, z_resolution=100, crossstream_dist=0.0)
cross_plane = fi.calculate_cross_plane(y_resolution=100, z_resolution=100, downstream_dist=500.0)

# Create the plots
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
wakeviz.visualize_cut_plane(horizontal_plane, ax=ax_list[0], title="Horizontal")
wakeviz.visualize_cut_plane(y_plane, ax=ax_list[1], title="Streamwise profile")
wakeviz.visualize_cut_plane(cross_plane, ax=ax_list[2], title="Spanwise profile")

wakeviz.show_plots()
