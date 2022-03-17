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


from time import perf_counter as timerpc
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


"""
04_sweep_wind_directions

This example demonstrates vectorization of wind direction.  
A vector of wind directions is passed to the intialize function 
and the powers of the two simulated turbines is computed for all
wind directions in one call

The power of both turbines for each wind direction is then plotted

"""

# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
fi_nodeflection = FlorisInterface("inputs/gch_nodeflection.yaml")

# Define a two turbine farm
wd_array = np.arange(0.0, 360.0, 1.0)
layout_x = 5 * 126.0 * np.arange(30)  # 30 turbines
layout_y = np.zeros_like(layout_x)

fi.reinitialize(
    layout = [layout_x, layout_y],
    wind_directions=wd_array
)
fi_nodeflection.reinitialize(
    layout=[layout_x, layout_y],
    wind_directions=wd_array,
)

# Define yaw angles
yaw_angles = np.zeros((len(wd_array), 1, len(layout_x))) 
# yaw_angles[:, :, 0] = 20.0  # First turbine misaligned

# Collect the turbine powers
start_time = timerpc()
fi.calculate_wake(yaw_angles=yaw_angles)
farm_powers = fi.get_farm_power().flatten() / 1E3 # In kW
t = timerpc() - start_time
print("Time spent (with defl. model): {:.3f} s.".format(t))

start_time = timerpc()
fi_nodeflection.calculate_wake(yaw_angles=yaw_angles)
farm_powers_nodeflection = fi_nodeflection.get_farm_power().flatten() / 1E3 # In kW
t = timerpc() - start_time
print("Time spent (without defl. model): {:.3f} s.".format(t))

# Plot
fig, ax = plt.subplots()
ax.plot(wd_array, farm_powers, label='With deflection model')
ax.plot(wd_array, farm_powers_nodeflection, label='Without deflection model')
ax.grid(True)
ax.legend()
ax.set_xlabel('Wind Direction (deg)')
ax.set_ylabel('Power (kW)')
plt.show()
