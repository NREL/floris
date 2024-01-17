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


# Generate a random time series of wind speeds, wind directions and turbulence intensities
N = 500
wd_array = wrap_360(270 * np.ones(N) + np.random.randn(N) * 20)
ws_array = np.clip(8 * np.ones(N) + np.random.randn(N) * 8, 3, 50)
ti_array = np.clip(0.1 * np.ones(N) + np.random.randn(N) * 0.05, 0, 0.25)

fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(7, 4))
ax = axarr[0]
ax.plot(wd_array, marker=".", ls="None")
ax.set_ylabel("Wind Direction")
ax = axarr[1]
ax.plot(ws_array, marker=".", ls="None")
ax.set_ylabel("Wind Speed")
ax = axarr[2]
ax.plot(ti_array, marker=".", ls="None")
ax.set_ylabel("Turbulence Intensity")


# Build the time series
time_series = TimeSeries(wd_array, ws_array)  # , turbulence_intensity=ti_array)

# Now build the wind rose
wind_rose = time_series.to_wind_rose()

# Plot the wind rose
fig, ax = plt.subplots(subplot_kw={"polar": True})
wind_rose.plot_wind_rose(ax=ax)

# Now set up a FLORIS model and initialize it using the time series and wind rose
fi = FlorisInterface("inputs/gch.yaml")
fi.reinitialize(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

fi_time_series = fi.copy()
fi_wind_rose = fi.copy()

fi_time_series.reinitialize(wind_data=time_series)
fi_wind_rose.reinitialize(wind_data=wind_rose)

fi_time_series.calculate_wake()
fi_wind_rose.calculate_wake()

time_series_power = fi_time_series.get_farm_power()
wind_rose_power = fi_wind_rose.get_farm_power()

time_series_aep = fi_time_series.get_farm_AEP_with_wind_data(time_series)
wind_rose_aep = fi_wind_rose.get_farm_AEP_with_wind_data(wind_rose)

print(f"AEP from TimeSeries {time_series_aep / 1e9:.2f} GWh")
print(f"AEP from WindRose {wind_rose_aep / 1e9:.2f} GWh")

plt.show()
