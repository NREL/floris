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

import floris.tools as wfct


maxSpeed = 8
minSpeed = 5

# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("../example_input_gauss_legacy.json")
# fi = wfct.floris_interface.FlorisInterface("../example_input_gauss.json")
fi.reinitialize_flow_field(wind_shear=0.0)

fig, axarr = plt.subplots(3, 3, figsize=(20, 5))

# No GROUND CORREECTION
fi.floris.farm.wake.velocity_model.flag_orig_vel_def = True

ax = axarr[0, 0]

# Calculate wake
fi.change_turbine([0], {"hub_height": 110})
fi.calculate_wake()

# Cross (y-normal) plane
cross_plane_110_n = fi.get_y_plane(y_loc=0)

wfct.visualization.visualize_cut_plane(
    cross_plane_110_n, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Hub Height = 110, no ground correction")

ax = axarr[1, 0]

# Calculate wake
fi.change_turbine([0], {"hub_height": 90})
fi.calculate_wake()

# Cross (y-normal) plane
cross_plane_90_n = fi.get_y_plane(y_loc=0)

wfct.visualization.visualize_cut_plane(
    cross_plane_90_n, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Hub Height = 90, no ground correction")

ax = axarr[2, 0]

# Calculate wake
fi.change_turbine([0], {"hub_height": 70})
fi.calculate_wake()

# Cross (y-normal) plane
cross_plane_70_n = fi.get_y_plane(y_loc=0)

wfct.visualization.visualize_cut_plane(
    cross_plane_70_n, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Hub Height = 70, no ground correction")

# Yes GROUND CORREECTION
fi.floris.farm.wake.velocity_model.flag_orig_vel_def = False

ax = axarr[0, 1]

# Calculate wake
fi.change_turbine([0], {"hub_height": 110})
fi.calculate_wake()

# Cross (y-normal) plane
cross_plane_110_y = fi.get_y_plane(y_loc=0)

wfct.visualization.visualize_cut_plane(
    cross_plane_110_y, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Hub Height = 110, with ground correction")

ax = axarr[1, 1]

# Calculate wake
fi.change_turbine([0], {"hub_height": 90})
fi.calculate_wake()

# Cross (y-normal) plane
cross_plane_90_y = fi.get_y_plane(y_loc=0)

wfct.visualization.visualize_cut_plane(
    cross_plane_90_y, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Hub Height = 90, with ground correction")

ax = axarr[2, 1]

# Calculate wake
fi.change_turbine([0], {"hub_height": 70})
fi.calculate_wake()

# Cross (y-normal) plane
cross_plane_70_y = fi.get_y_plane(y_loc=0)

wfct.visualization.visualize_cut_plane(
    cross_plane_70_y, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Hub Height = 70, with ground correction")


# Get the differences
minSpeed = -0.5
maxSpeed = 0.5

diff110 = wfct.cut_plane.subtract(cross_plane_110_y, cross_plane_110_n)
diff90 = wfct.cut_plane.subtract(cross_plane_90_y, cross_plane_90_n)
diff70 = wfct.cut_plane.subtract(cross_plane_70_y, cross_plane_70_n)

ax = axarr[0, 2]
im = wfct.visualization.visualize_cut_plane(
    diff110, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Difference")

ax = axarr[1, 2]
im = wfct.visualization.visualize_cut_plane(
    diff90, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Difference")

ax = axarr[2, 2]
im = wfct.visualization.visualize_cut_plane(
    diff70, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed
)
ax.set_title("Difference")
# cbar = fig.colorbar(im, ax =ax)

plt.show()
