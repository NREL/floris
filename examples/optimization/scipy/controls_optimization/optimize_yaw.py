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
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.yaw import YawOptimization
import numpy as np
import os

print('Running FLORIS with no yaw...')
# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, '../../../example_input.json')
)

# Set turbine locations to 3 turbines in a row
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 7 * D, 14 * D]
layout_y = [0, 0, 0]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
fi.calculate_wake()

# Initial power output
power_initial = fi.get_farm_power()

# =============================================================================
print('Plotting the FLORIS flowfield...')
# =============================================================================

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Baseline Case for U = 8 m/s, Wind Direction = 270$^\circ$')

# =============================================================================
print('Finding optimal yaw angles in FLORIS...')
# =============================================================================
# Set bounds for allowable wake steering
min_yaw = 0.0
max_yaw = 25.0

# Instantiate the Optimization object
yaw_opt = YawOptimization(fi,
                          minimum_yaw_angle=min_yaw,
                          maximum_yaw_angle=max_yaw)

# Perform optimization
yaw_angles = yaw_opt.optimize()

print('==========================================')
print('yaw angles = ')
for i in range(len(yaw_angles)):
    print('Turbine ', i, '=', yaw_angles[i], ' deg')

# Assign yaw angles to turbines and calculate wake
fi.calculate_wake(yaw_angles=yaw_angles)
power_opt = fi.get_farm_power()

print('==========================================')
print('Total Power Gain = %.1f%%' %
      (100. * (power_opt - power_initial) / power_initial))
print('==========================================')
# =============================================================================
print('Plotting the FLORIS flowfield with yaw...')
# =============================================================================

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title(
    'Optimal Wake Steering for U = 8 m/s, Wind Direction = 270$^\circ$')
plt.show()
