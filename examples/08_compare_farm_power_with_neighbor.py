# Copyright 2022 NREL

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
import pandas as pd
from floris.tools import FlorisInterface
import matplotlib.pyplot as plt

"""
This example demonstrates how to use turbine_wieghts to define a set of turbines belonging to a neighboring farm which
impacts the power production of the farm under consideration via wake losses, but whose own power production is not
considered in farm power / aep production

The use of neighboring farms in the context of wake steering design is considered in example examples/10_optimize_yaw_with_neighboring_farm.py
"""


# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2

# Define a 4 turbine farm turbine farm
D = 126.
layout_x = np.array([0, D*6, 0, D*6])
layout_y = [0, 0, D*3, D*3]
fi.reinitialize(layout = [layout_x, layout_y])

# Define a simple wind rose with just 1 wind speed
wd_array = np.arange(0,360,4.)
fi.reinitialize(wind_directions=wd_array, wind_speeds=[8.])


# Calculate
fi.calculate_wake()

# Collect the farm power
farm_power_base = fi.get_farm_power() / 1E3 # In kW

# Add a neighbor to the east
layout_x = np.array([0, D*6, 0, D*6, D*12, D*15, D*12, D*15])
layout_y = np.array([0, 0, D*3, D*3, 0, 0, D*3, D*3])
fi.reinitialize(layout = [layout_x, layout_y])

# Define the weights to exclude the neighboring farm from calcuations of power
turbine_weights = np.zeros(len(layout_x), dtype=int)
turbine_weights[0:4] = 1.0

# Calculate
fi.calculate_wake()

# Collect the farm power with the neightbor
farm_power_neighbor = fi.get_farm_power(turbine_weights=turbine_weights) / 1E3 # In kW

# Show the farms
fig, ax = plt.subplots()
ax.scatter(layout_x[turbine_weights==1],layout_y[turbine_weights==1], color='k',label='Base Farm')
ax.scatter(layout_x[turbine_weights==0],layout_y[turbine_weights==0], color='r',label='Neighboring Farm')
ax.legend()

# Plot the power difference
fig, ax = plt.subplots()
ax.plot(wd_array,farm_power_base,color='k',label='Farm Power (no neighbor)')
ax.plot(wd_array,farm_power_neighbor,color='r',label='Farm Power (neighboring farm due east)')
ax.grid(True)
ax.legend()
ax.set_xlabel('Wind Direction (deg)')
ax.set_ylabel('Power (kW)')
plt.show()
