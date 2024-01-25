# Copyright 2024 NREL

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

from floris.tools import (
    FlorisInterface,
    TimeSeries,
    WindRose,
)
from floris.utilities import wrap_360


"""
Demonstrate the new behavior in T4 where TI is an array rather than a float.
Set up an array of two turbines and sweep TI while hold wd/ws constant.
Use the TimeSeries object to drive the FLORIS calculations.
"""


# Generate a random time series of wind speeds, wind directions and turbulence intensities
N = 50
wd_array = 270.0 * np.ones(N)
ws_array = 8.0 * np.ones(N)
ti_array = np.linspace(0.03, 0.2, N)


# Build the time series
time_series = TimeSeries(wd_array, ws_array, turbulence_intensities=ti_array)


# Now set up a FLORIS model and initialize it using the time
fi = FlorisInterface("inputs/gch.yaml")
fi.reinitialize(layout_x=[0, 500.0], layout_y=[0.0, 0.0], wind_data=time_series)
fi.calculate_wake()
turbine_power = fi.get_turbine_powers()

fig, axarr = plt.subplots(5, 1, sharex=True, figsize=(5, 9))
ax = axarr[0]
ax.plot(wd_array, color="k")
ax.set_ylabel("Wind Direction")
ax = axarr[1]
ax.plot(ws_array, color="k")
ax.set_ylabel("Wind Speed")
ax = axarr[2]
ax.plot(ti_array, color="k")
ax.set_ylabel("Turbulence Intensity")
ax = axarr[3]
ax.plot(turbine_power[:, 0], color="k")
ax.set_ylabel("Front Turbine")
ax = axarr[4]
ax.plot(turbine_power[:, 1], color="k")
ax.set_ylabel("Rear Turbine")

for ax in axarr:
    ax.grid(True)

plt.show()
