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
fi.reinitialize(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

# Single wind speed and wind direction
print("\n========================= Single Wind Direction and Wind Speed =========================")

# Get the turbine powers assuming 1 wind direction and speed
fi.reinitialize(wind_directions=[270.0], wind_speeds=[8.0])

# Set the yaw angles to 0
yaw_angles = np.zeros([1, 2])  # 1 wind direction and speed, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)

# Get the turbine powers
turbine_powers = fi.get_turbine_powers() / 1000.0

print("The turbine power matrix should be of dimensions 1 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

# Single wind speed and multiple wind directions
print("\n========================= Single Wind Direction and Multiple Wind Speeds ===============")

wind_speeds = np.array([8.0, 9.0, 10.0])
wind_directions = np.array([270.0, 270.0, 270.0])

fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)
yaw_angles = np.zeros([3, 2])  # 3 wind directions/ speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 3 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

# Multiple wind speeds and multiple wind directions
print("\n========================= Multiple Wind Directions and Multiple Wind Speeds ============")

# To consider each combination, this needs to be broadcast out in advance

wind_speeds = np.tile([8.0, 9.0, 10.0], 3)
wind_directions = np.repeat([260.0, 270.0, 280.0], 3)

fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)
yaw_angles = np.zeros([9, 2])  # 9 wind directions/ speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 9 WD/WS X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)
