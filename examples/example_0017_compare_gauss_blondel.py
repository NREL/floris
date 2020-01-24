# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("example_input.json")

# Get HH and D
HH = fi.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi.floris.farm.turbines[0].rotor_diameter

# Go to a signle turbine
fi.reinitialize_flow_field(layout_array=([0],[0]))

# Calculate wake
fi.calculate_wake()

# Define to cut across profile
step_size = 5
y_points = np.arange(-100,100+step_size,step_size)
x_points = np.ones_like(y_points) * 3 * D
z_points = np.ones_like(x_points) * HH

# Get the values
flow_points = fi.get_set_of_points(x_points,y_points,z_points)

print(flow_points)

# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_hor_plane()

# Plot and show
fig, axarr = plt.subplots(1,2)

ax = axarr[0]
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.plot(x_points,y_points,'r',lw=3)

ax = axarr[1]
ax.plot(flow_points.y,flow_points.u)

plt.show()
