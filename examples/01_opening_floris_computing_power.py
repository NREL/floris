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


import numpy as np

from floris.tools import FlorisInterface


"""
This example creates a FLORIS instance
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")

# Convert to a simple two turbine layout
fi.reinitialize(layout_x=[0, 500.], layout_y=[0., 0.])

# Single wind speed and wind direction
print('\n========================= Single Wind Direction and Wind Speed =========================')

# Get the turbine powers assuming 1 wind speed and 1 wind direction
fi.reinitialize(wind_directions=[270.], wind_speeds=[8.0])

# Set the yaw angles to 0
yaw_angles = np.zeros([1,1,2]) # 1 wind direction, 1 wind speed, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)

# Get the turbine powers
turbine_powers = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 1 WD X 1 WS X 2 Turbines')
print(turbine_powers)
print("Shape: ",turbine_powers.shape)

# Single wind speed and wind direction
print('\n========================= Single Wind Direction and Multiple Wind Speeds ===============')


wind_speeds = np.array([8.0, 9.0, 10.0])
fi.reinitialize(wind_speeds=wind_speeds)
yaw_angles = np.zeros([1,3,2]) # 1 wind direction, 3 wind speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 1 WD X 3 WS X 2 Turbines')
print(turbine_powers)
print("Shape: ",turbine_powers.shape)

# Single wind speed and wind direction
print('\n========================= Multiple Wind Directions and Multiple Wind Speeds ============')

wind_directions = np.array([260., 270., 280.])
wind_speeds = np.array([8.0, 9.0, 10.0])
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)
yaw_angles = np.zeros([1,3,2]) # 1 wind direction, 3 wind speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers()/1000.
print('The turbine power matrix should be of dimensions 3 WD X 3 WS X 2 Turbines')
print(turbine_powers)
print("Shape: ",turbine_powers.shape)
