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
import numpy as np

from floris.tools import FlorisInterface

# from floris.tools.visualization import visualize_cut_plane

"""
00_getting_started

This initial example creates a FLORIS instance 
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")

# Convert to a simple two turbine layout
fi.reinitialize( layout=( [0, 500.], [0., 0.] ) )

# Get the turbine powers assuming 1 wind speed and 1 wind direction
fi.reinitialize(wind_directions=[270.], wind_speeds=[8.0])

# Set the yaw angles to 0
yaw_angles = np.zeros([1,1,2]) # 1 wind direction, 1 wind speed, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)

# Get the turbine powers
turbine_powers = fi.get_turbine_powers()/1000.
print('===========')
print('The turbine power matrix should be of dimensions 1 WD X 1 WS X 2 Turbines')
print(turbine_powers)
print('===========')

# Now apply 3 wind speeds
wind_speeds = np.array([8.0, 9.0, 10.0])
fi.reinitialize( wind_speeds=wind_speeds)
yaw_angles = np.zeros([1,3,2]) # 1 wind direction, 3 wind speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers()/1000.
print('===========')
print('The turbine power matrix should be of dimensions 1 WD X 3 WS X 2 Turbines')
print(turbine_powers)
print('===========')

# Make a small plot
fig, ax = plt.subplots()
ax.plot(wind_speeds,turbine_powers[:,:,0].flatten(), color='k',label='Turbine 0' )
ax.plot(wind_speeds,turbine_powers[:,:,1].flatten(), color='r',label='Turbine 1' )
ax.grid()
ax.legend()
ax.set_ylabel('Power (kW)')
ax.set_xlabel('Wind Speed (m/s)')
plt.show()

