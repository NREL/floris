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
This example demonstrates the ability of FLORIS to shut down some turbines
during a simulation.
"""

# Initialize the FLORIS interface
fi = FlorisInterface("inputs/gch.yaml")

# Data
# ------------------------------------------

# Consider a wind farm of 3 aligned wind turbines
layout = np.array([[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0]])

# Run the computations for 2 identical wind data
wind_data = np.array([[270.0, 7.0], [270.0, 7.0]])

# Keep turbines aligned
yaw_angles = np.zeros((wind_data.shape[0], 1, layout.shape[0]))

# Shut down the first 2 turbines for the second wind data
turbines_off = np.array([[[False, False, False]], [[True, True, False]]])

# Simulation
# ------------------------------------------

# Reinitialize flow field
fi.reinitialize(layout_x=layout[:, 0],
                layout_y=layout[:, 1],
                wind_directions=wind_data[:, 0],
                wind_speeds=wind_data[:, 1],
                time_series=True)

# Compute wakes
fi.calculate_wake(turbines_off=turbines_off, yaw_angles=yaw_angles)

# Results
# ------------------------------------------

# Get powers and effective wind speeds
turbine_powers = fi.get_turbine_powers()
turbine_powers = np.round(turbine_powers * 1e-6, decimals=2)
effective_wind_speeds = fi.turbine_average_velocities

# Print results
print("wind_data")
print(wind_data)
print("\nturbines_off")
print(turbines_off)
print("\nturbine_powers [MWs]")
print(turbine_powers)
print("\neffective_wind_speeds [m/s]")
print(effective_wind_speeds)
