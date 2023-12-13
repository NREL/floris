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


"""
This example demonstrates running FLORIS given a time series
of wind direction and wind speed combinations.
"""

# Initialize FLORIS to simple 4 turbine farm
fi = FlorisInterface("inputs/gch.yaml")

# Convert to a simple two turbine layout
fi.reinitialize(layout_x=[0, 500.], layout_y=[0., 0.])

# Create a fake time history where wind speed steps in the middle while wind direction
# Walks randomly
time = np.arange(0, 120, 10.) # Each time step represents a 10-minute average
ws = np.ones_like(time) * 8.
ws[int(len(ws) / 2):] = 9.
wd = np.ones_like(time) * 270.

for idx in range(1, len(time)):
    wd[idx] = wd[idx - 1] + np.random.randn() * 2.


# Now intiialize FLORIS object to this history using time_series flag
fi.reinitialize(wind_directions=wd, wind_speeds=ws)

# Collect the powers
fi.calculate_wake()
turbine_powers = fi.get_turbine_powers() / 1000.

# Show the dimensions
num_turbines = len(fi.layout_x)
print(
    f'There are {len(time)} time samples, and {num_turbines} turbines and '
    f'so the resulting turbine power matrix has the shape {turbine_powers.shape}.'
)


fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(7,8))

ax = axarr[0]
ax.plot(time, ws, 'o-')
ax.set_ylabel('Wind Speed (m/s)')
ax.grid(True)

ax = axarr[1]
ax.plot(time, wd, 'o-')
ax.set_ylabel('Wind Direction (Deg)')
ax.grid(True)

ax = axarr[2]
for t in range(num_turbines):
    ax.plot(time,turbine_powers[:, t], 'o-', label='Turbine %d' % t)
ax.legend()
ax.set_ylabel('Turbine Power (kW)')
ax.set_xlabel('Time (minutes)')
ax.grid(True)

plt.show()
