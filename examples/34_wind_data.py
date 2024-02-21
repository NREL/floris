
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import (
    FlorisInterface,
    TimeSeries,
    WindRose,
)
from floris.utilities import wrap_360


"""
This example is meant to be temporary and may be updated by a later pull request. Before we
release v4, we intend to propagate the TimeSeries and WindRose objects through the other relevant
examples, and change this example to demonstrate more advanced (as yet, not implemented)
functionality of the WindData objects (such as electricity pricing etc).
"""


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
time_series = TimeSeries(wd_array, ws_array, turbulence_intensities=ti_array)

# Now build the wind rose
wind_rose = time_series.to_wind_rose()

# Plot the wind rose
fig, ax = plt.subplots(subplot_kw={"polar": True})
wind_rose.plot_wind_rose(ax=ax)

# Now set up a FLORIS model and initialize it using the time series and wind rose
fi = FlorisInterface("inputs/gch.yaml")
fi.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

fi_time_series = fi.copy()
fi_wind_rose = fi.copy()

fi_time_series.set(wind_data=time_series)
fi_wind_rose.set(wind_data=wind_rose)

fi_time_series.run()
fi_wind_rose.run()

time_series_power = fi_time_series.get_farm_power()
wind_rose_power = fi_wind_rose.get_farm_power()

time_series_aep = fi_time_series.get_farm_AEP_with_wind_data(time_series)
wind_rose_aep = fi_wind_rose.get_farm_AEP_with_wind_data(wind_rose)

print(f"AEP from TimeSeries {time_series_aep / 1e9:.2f} GWh")
print(f"AEP from WindRose {wind_rose_aep / 1e9:.2f} GWh")

plt.show()
