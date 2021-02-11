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


# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Calculate wake
layout_x = [0, 800, 1600]
layout_y = [0, 60, 60]
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))
fi.calculate_wake()
power_init = fi.get_farm_power()
print(fi.get_turbine_power())

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

# change the tilt angle
fi.reinitialize_flow_field()
fi.calculate_wake(yaw_angles=[0.0, 0.0, 0.0], tilt_angles=[25.0, 0.0, 0.0])
power_tilt = fi.get_farm_power()
print(fi.get_turbine_power())

print("Power difference = ", 100 * (power_tilt - power_init) / power_init)

# Plot and show
y_plane = fi.get_y_plane(0.0)
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(y_plane, ax=ax)

plt.show()
