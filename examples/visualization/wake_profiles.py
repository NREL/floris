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


D = 126.0
HH = 90.0

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Single turbine at 0,0
fi.reinitialize_flow_field(layout_array=([0], [0]), wind_speed=8.0, wind_shear=0.0)

# Calculate wake
fi.calculate_wake()

# Get a cross plane at 7D
cross_plane = fi.get_cross_plane(x_loc=7 * D)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(cross_plane, ax=ax)
ax.set_title("Cross Plane")


# Show the velocity profile using a rotor sweep
x1_locs, v_array = wfct.cut_plane.wind_speed_profile(
    cross_plane, D / 2, HH, resolution=100, x1_locs=None
)
fig, ax = plt.subplots()
ax.plot(x1_locs, v_array, color="k")
ax.grid(True)
ax.set_title("Rotor-averaged velocity profile")
ax.set_ylabel("Rotor-averaged wind speed (m/s)")
ax.set_xlabel("Center of rotor disk (m)")

plt.show()
