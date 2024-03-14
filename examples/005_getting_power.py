"""Example 5: Getting Turbine and Farm Power

After setting the FlorisModel and running, the next step is typically to get the power output
of the turbines.

"""

import matplotlib.pyplot as plt
import numpy as np
import yaml

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
    WindTIRose,
)


# Initialize FLORIS with the given input file via FlorisModel
fmodel = FlorisModel("inputs/gch.yaml")

# Set to a 3-turbine layout
fmodel.set(layout_x=[0, 126*5, 126*10], layout_y=[0, 0, 0])

######################################################
# Using TimeSeries
######################################################

# Set up a time series in which the wind speed and TI are constant but the wind direction
# sweeps the range from 250 to 290 degrees
wind_directions = np.arange(250, 290, 1.0)
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=9.9, turbulence_intensities=0.06
)
fmodel.set(wind_data=time_series)

# Run the model
fmodel.run()

# Get the turbine powers
turbine_powers = fmodel.get_turbine_powers()

# Turbines powers will have shape (n_findex, n_tindex) where n_findex is the number of unique
# wind conditions and n_tindex is the number of turbines in the farm
print(f"Turbine power has shape {turbine_powers.shape}")

# It is also possible to get the farm power directly
farm_power = fmodel.get_farm_power()

# Farm power has length n_findex, and is the sum of the turbine powers
print(f"Farm power has shape {farm_power.shape}")

# It's possible to get these powers with wake losses disabled, this can be useful
# for computing total wake losses
fmodel.run_no_wake()
farm_power_no_wake = fmodel.get_farm_power()

# Plot the results
fig, axarr = plt.subplots(1, 3, figsize=(15, 5))

# Plot the turbine powers
ax = axarr[0]
for i in range(turbine_powers.shape[1]):
    ax.plot(wind_directions, turbine_powers[:, i] / 1e3, label=f"Turbine {i+1}  ")
ax.set_xlabel("Wind Direction (deg)")
ax.set_ylabel("Power (kW)")
ax.grid(True)
ax.legend()
ax.set_title("Turbine Powers")

# Plot the farm power
ax = axarr[1]
ax.plot(wind_directions, farm_power / 1e3, label='Farm Power With Wakes', color='k')
ax.plot(wind_directions, farm_power_no_wake / 1e3, label='Farm Power No Wakes', color='r')
ax.set_xlabel("Wind Direction (deg)")
ax.set_ylabel("Power (kW)")
ax.grid(True)
ax.legend()
ax.set_title("Farm Power")

# Plot the percent wake losses
ax = axarr[2]
percent_wake_losses = 100 * (farm_power_no_wake - farm_power) / farm_power_no_wake
ax.plot(wind_directions, percent_wake_losses, label='Percent Wake Losses', color='k')
ax.set_xlabel("Wind Direction (deg)")
ax.set_ylabel("Percent Wake Losses")
ax.grid(True)
ax.legend()
ax.set_title("Percent Wake Losses")


######################################################
# Using WindRose
######################################################

# When running FLORIS using a wind rose, the wind data is held in a
# wind_directions x wind_speeds table
# form, which is unpacked into a 1D array within the FlorisModel.
#  Additionally wind direction and
# wind speed combinations which have 0 frequency are not computed, unless the user specifies
# the `compute_zero_freq_occurrence=True` option in the WindRose constructor.

# When calculating AEP, the bins can be combined automatically

#TODO: Revist this section after https://github.com/NREL/floris/pull/844 is merged

plt.show()
