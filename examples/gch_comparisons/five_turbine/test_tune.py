# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation


import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np


print('Running FLORIS with no yaw...')
# Instantiate the FLORIS object

initial = np.linspace(0.1,0.9,4)
constant = np.linspace(0.1,0.9,4)
ai = np.linspace(0.1,0.9,4)
downstream = np.linspace(0.1,0.9,4)

fi = wfct.floris_interface.FlorisInterface("../../example_input.json")

# Set turbine locations to 3 turbines in a row
D = fi.floris.farm.turbines[0].rotor_diameter

l_x = [0,6*D,12*D,18*D,24*D]
# l_x = [0,7*D,14*D]
l_y = [0,0,0,0,0]

# fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),wind_direction=wind_direction)
fi.reinitialize_flow_field(layout_array=(l_x, l_y),wind_direction=270)
fi.calculate_wake()

# Initial power output
power_initial = fi.get_farm_power()
print('Initial farm power = ', power_initial)
for i in range(len(l_x)):
    print('Turbine ', i, ' velocity = ', fi.floris.farm.turbines[i].average_velocity, fi.floris.farm.turbines[i].power/(10**3))


# =============================================================================
print('Finding optimal yaw angles in FLORIS...')
# =============================================================================
# Set bounds for allowable wake steering
min_yaw = 0.0
max_yaw = 25.0

# Instantiate the Optimization object
# yaw_opt = YawOptimization(fi,
#                                minimum_yaw_angle=min_yaw,
#                                maximum_yaw_angle=max_yaw)

# Perform optimization
# yaw_angles = yaw_opt.optimize()
# yaw_angles = [24,24,22,16,0]
yaw_angles = [25,25,25,0,0]
# yaw_angles = [20,20,0]

fi.reinitialize_flow_field()
print('==========================================')
fi.calculate_wake(yaw_angles=yaw_angles)

for i in range(len(l_x)):
    print('Turbine ', i, ' velocity = ', fi.floris.farm.turbines[i].average_velocity, fi.floris.farm.turbines[i].power/(10**3))

# Assign yaw angles to turbines and calculate wake
power_opt = fi.get_farm_power()
print('Power initial = ', power_initial/(10**3))
print('Power optimal = ', power_opt/(10**3))

print('==========================================')
print('Total Power Gain = %.1f%%' %
      (100.*(power_opt - power_initial)/power_initial))
print('==========================================')

# =============================================================================
print('Finding optimal yaw angles in FLORIS low ti...')
# =============================================================================

# Set bounds for allowable wake steering
min_yaw = 0.0
max_yaw = 25.0

# Instantiate the Optimization object
fi.reinitialize_flow_field(turbulence_intensity=0.065)
fi.calculate_wake(yaw_angles=np.zeros(len(l_x)))

# Initial power output
power_initial = fi.get_farm_power()

# Perform optimization
# yaw_angles = yaw_opt.optimize()
yaw_angles = [25,25,22,18,0]
# yaw_angles = [25,25,25,0,0]
# yaw_angles = [20,20,0]
print('==========================================')
fi.reinitialize_flow_field()
fi.calculate_wake(yaw_angles=yaw_angles)

for i in range(len(l_x)):
    print('Turbine ', i, ' velocity = ', fi.floris.farm.turbines[i].average_velocity, fi.floris.farm.turbines[i].power/(10**3))

# Assign yaw angles to turbines and calculate wake
power_opt = fi.get_farm_power()
print('Power initial = ', power_initial/(10**3))
print('Power optimal = ', power_opt/(10**3))

print('==========================================')
print('Total Power Gain = %.1f%%' %
      (100.*(power_opt - power_initial)/power_initial))
print('==========================================')

## For tuning TI model

SB =  [2419000.,  915100.,  945100., 1046200., 1037700., 1077700.]
SOC =  [2046400., 1062300., 1239500., 1353700., 1421400., 1761700.]
layout_x =  [1145.6, 1791.2, 2436.8, 3082.4, 3728.,  4373.6]
layout_y = [2436.8, 2436.8, 2436.8, 2436.8, 2436.8, 2436.8]
yaw = [23.6, 23.2, 21.,  18.1, 13.9,  0. ]

# no yaw
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y),turbulence_intensity=0.09)
fi.calculate_wake(yaw_angles=np.zeros(len(layout_x)))
GCH_Base = fi.get_turbine_power()

# yaw
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y),turbulence_intensity=0.09)
fi.calculate_wake(yaw_angles=yaw)
GCH_opt = fi.get_turbine_power()

plt.figure()
turb = np.linspace(0,5,6)
plt.plot(turb,SB,'ko--',label='sowfa_base')
plt.plot(turb,GCH_Base,'ro--',label='gch_base')
plt.plot(turb,SOC,'ko-',label='sowfa_opt')
plt.plot(turb,GCH_opt,'ro-',label='gch_opt')
plt.grid()
plt.legend()
plt.title('Baseline Power (middle row)')

# # Get horizontal plane at default height (hub-height)
# hor_plane = fi.get_hor_plane()
#
# # Plot and show
# fig, ax = plt.subplots()
# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
plt.show()



