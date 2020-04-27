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
 

# This example optimization takes a 1x5 array of turbines at an initial spacing
# of 9 rotor diameters and works to compress that spacing in the streamwise (x)
# direction, while working to maintain the original energy output by leveraging
# wake steering. It is meant to be an illustrative example of some of the
# benefits of wake steering.

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris.tools.optimization.scipy.yaw import YawOptimization
import numpy as np
import os

print('Running FLORIS with no yaw...')
# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, '../../../example_input.json')
)

# Set turbine locations to a 2 turbine array
D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = [0, 5*D]
layout_y = [0, 0]
fi.reinitialize_flow_field(
    layout_array=(layout_x, layout_y),
    wind_direction = [273.0]
)
fi.calculate_wake()

unc_options={'std_wd': 4.95, 'std_yaw': 1.75,'pmf_res': 1.0, 'pdf_cutoff': 0.99}

# Initial power output without uncertainty
power_initial = fi.get_farm_power()

# Initial power output with uncertainty
power_initial_unc = fi.get_farm_power(include_unc=True,
                        unc_options=unc_options)

# =============================================================================
print('Plotting the FLORIS flowfield...')
# =============================================================================

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(height=fi.floris.farm.turbines[0].hub_height)

# Plot and show
fig, ax = plt.subplots(figsize=(7.0, 5.0))
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Baseline Case for U = 8 m/s, Wind Direction = 273$^\circ$')

# =============================================================================
print('Finding optimal yaw angles in FLORIS...')
# =============================================================================
# Set bounds for allowable wake steering
min_yaw = 0.0
max_yaw = 25.0

# Without uncertainty
# Instantiate the Optimization object
yaw_opt = YawOptimization(
    fi,
    minimum_yaw_angle=min_yaw, 
    maximum_yaw_angle=max_yaw
)

# Perform optimization
yaw_angles = yaw_opt.optimize()

# With Uncertainty
# Instantiate the Optimization object
yaw_opt = YawOptimization(
    fi,
    minimum_yaw_angle=min_yaw, 
    maximum_yaw_angle=max_yaw,
    include_unc=True,
    unc_options=unc_options
)

# Perform optimization
yaw_angles_unc = yaw_opt.optimize()

print('==========================================')
print('yaw angles without uncertainty = ')
for i in range(len(yaw_angles)):
    print('Turbine ', i, '=', yaw_angles[i], ' deg')
print('robust yaw angles with uncertainty = ')
for i in range(len(yaw_angles_unc)):
    print('Turbine ', i, '=', yaw_angles_unc[i], ' deg')

# Assign yaw angles to turbines and calculate wake
fi.calculate_wake(yaw_angles=yaw_angles)
power_opt = fi.get_farm_power()
power_opt_unc = fi.get_farm_power(
    include_unc=True,
    unc_options=unc_options
)
fi.calculate_wake(yaw_angles=yaw_angles_unc)
power_opt_unc_robust = fi.get_farm_power(
    include_unc=True,
    unc_options=unc_options
)

print('==========================================')
print('Total Power Gain without Uncertainty = %.1f%%' %
      (100.*(power_opt - power_initial)/power_initial))
print('Total Power Gain with Uncertainty using Original Yaw Angles = %.1f%%' %
      (100.*(power_opt_unc - power_initial_unc)/power_initial_unc))
print('Total Power Gain with Uncertainty using Robust Yaw Angles = %.1f%%' %
      (100.*(power_opt_unc_robust - power_initial_unc)/power_initial_unc))
print('==========================================')
# =============================================================================
print('Plotting the FLORIS flowfield with yaw...')
# =============================================================================

# Initialize the horizontal cut without uncertainty
fi.calculate_wake(yaw_angles=yaw_angles)
hor_plane = fi.get_hor_plane(
    x_resolution=400,
    y_resolution=100,
    height=fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots(figsize=(7.0, 5.0))
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Optimal Wake Steering without Uncertainty for U = 8 m/s, ' + \
             'Wind Direction = 273$^\circ$')

# Initialize the horizontal cut for robust wake steering with uncertainty
fi.calculate_wake(yaw_angles=yaw_angles_unc)
hor_plane = fi.get_hor_plane(
    x_resolution=400,
    y_resolution=100,
    height=fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots(figsize=(7.0, 5.0))
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Optimal Wake Steering with Uncertainty for U = 8 m/s, ' + \
             'Wind Direction = 273$^\circ$')
plt.show()
