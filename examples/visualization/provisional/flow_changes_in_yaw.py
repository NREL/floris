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
cut_plane_base_5 = fi.get_cross_plane(5*D)
cut_plane_base_7 = fi.get_cross_plane(7*D)

# Calculate yaw
fi.calculate_wake(yaw_angles=[25])
cut_plane_yaw_5 = fi.get_cross_plane(5*D)
cut_plane_yaw_7 = fi.get_cross_plane(7*D)

# Get the difference planes
cut_plane_diff_5 = cp.subtract(cut_plane_yaw_5,cut_plane_base_5)
cut_plane_diff_7 = cp.subtract(cut_plane_yaw_7,cut_plane_base_7)


# Plot and show
fig, axarr = plt.subplots(2,3,figsize=(15,5))

ax = axarr[0,0]
wfct.visualization.visualize_cut_plane(cut_plane_base_5, ax=ax,minSpeed=4,maxSpeed=8)
ax.set_title("Baseline, 5D")

ax = axarr[1,0]
wfct.visualization.visualize_cut_plane(cut_plane_base_7, ax=ax,minSpeed=4,maxSpeed=8)
ax.set_title("Baseline, 7D")

ax = axarr[0,1]
wfct.visualization.visualize_cut_plane(cut_plane_yaw_5, ax=ax,minSpeed=4,maxSpeed=8)
ax.set_title("Yaw, 5D")

ax = axarr[1,1]
wfct.visualization.visualize_cut_plane(cut_plane_yaw_7, ax=ax,minSpeed=4,maxSpeed=8)
ax.set_title("Yaw, 7D")

ax = axarr[0,2]
wfct.visualization.visualize_cut_plane(cut_plane_diff_5, ax=ax,minSpeed=-1,maxSpeed=1)
ax.set_title("Difference, 5D")

ax = axarr[1,2]
wfct.visualization.visualize_cut_plane(cut_plane_diff_7, ax=ax,minSpeed=-1,maxSpeed=1)
ax.set_title("Difference, 7D")

# Reverse axis  making the view upstream looking down
for ax in axarr.flatten():
    wfct.visualization.reverse_cut_plane_x_axis_in_plot(ax)

plt.show()
