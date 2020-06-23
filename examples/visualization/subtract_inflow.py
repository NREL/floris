# Copyright 2020 NREL

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

import floris.tools as wfct
import floris.tools.cut_plane as cp


# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Single turbine at 0,0
fi.reinitialize_flow_field(layout_array=([0], [0]))

# Calculate wake
fi.calculate_wake()

# Grab some cross planes
D = 126
cut_plane_base_5 = fi.get_cross_plane(5 * D)
cut_plane_base_in = fi.get_cross_plane(-5 * D)


# Get the difference planes
cut_plane_diff = cp.subtract(cut_plane_base_5, cut_plane_base_in)


# Plot and show
fig, axarr = plt.subplots(3, 1, figsize=(7, 10))

ax = axarr[0]
wfct.visualization.visualize_cut_plane(cut_plane_base_5, ax=ax, minSpeed=4, maxSpeed=8)
ax.set_title("Baseline, 5D")

ax = axarr[1]
wfct.visualization.visualize_cut_plane(cut_plane_base_in, ax=ax, minSpeed=4, maxSpeed=8)
ax.set_title("Baseline, Inflow")

ax = axarr[2]
wfct.visualization.visualize_cut_plane(cut_plane_diff, ax=ax, minSpeed=-2, maxSpeed=2)
ax.set_title("5D - INFLOW")

# Reverse axis  making the view upstream looking down
for ax in axarr.flatten():
    wfct.visualization.reverse_cut_plane_x_axis_in_plot(ax)

plt.show()
