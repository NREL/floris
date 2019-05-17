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
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris.tools.optimization import optimize_yaw
import numpy as np


print('Running FLORIS with no yaw...')
fi = wfct.floris_utilities.FlorisInterface("example_input.json")

# set turbine locations to 3 turbines in a row
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
fi.calculate_wake()

# initial power output
power_initial = np.sum(fi.get_turbine_power())

# ================================================================================
print('Plotting the FLORIS flowfield...')
# ================================================================================

# Initialize the horizontal cut
hor_plane = wfct.cut_plane.HorPlane(
    fi.get_hub_height_flow_data(),
    fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

# ================================================================================
print('Finding optimal yaw angles in FLORIS...')
# ================================================================================
# set bounds for allowable wake steering
min_yaw = 0.0
max_yaw = 25.0
yaw_angles = optimize_yaw(fi, min_yaw, max_yaw)

print('yaw angles = ')
for i in range(len(yaw_angles)):
    print('Turbine ', i, '=', yaw_angles[i], ' deg')

# assign yaw angles to turbines and calculate wake
fi.calculate_wake(yaw_angles=yaw_angles)
power_opt = np.sum(fi.get_turbine_power())

print('==========================================')
print('Total Power Gain = %.1f%%' %
      (100.*(power_opt - power_initial)/power_initial))
print('==========================================')
# ================================================================================
print('Plotting the FLORIS flowfield with yaw...')
# ================================================================================

# Initialize the horizontal cut
hor_plane = wfct.cut_plane.HorPlane(
    fi.get_hub_height_flow_data(),
    fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
plt.show()
