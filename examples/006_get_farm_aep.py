"""Example 6: Getting AEP

AEP is annual energy production and can is typically a weighted sum over farm power.  This
example demonstrates how to calculate the AEP

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)


# Initialize FLORIS with the given input file via FlorisModel
fmodel = FlorisModel("inputs/gch.yaml")


# Set to a 5-turbine layout
fmodel.set(layout_x=[0, 126 * 5, 126 * 10, 126 * 15, 126 * 20], layout_y=[0, 0, 0, 0, 0])

# Using TimeSeries

# In the case of time series data, although not required, the typical assumption is
# that each time step is equally likely.

# Randomly generated a time series with time steps = 365 * 24
N = 365 * 24
wind_directions = np.random.uniform(0, 360, N)
wind_speeds = np.random.uniform(5, 25, N)

# Set up a time series
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=0.06
)

# Note that the AEP functions run the model
# So it is not necessary to call run()
fmodel.set(wind_data=time_series)

aep = fmodel.get_farm_AEP_with_wind_data(time_series)

# Note this is equivalent to the following
aep_b = fmodel.get_farm_AEP(time_series.unpack_freq())

print(f"AEP from time series: {aep}, and re-computed AEP: {aep_b}")

# Using WindRose

# Assume a provided wind rose of frequency by wind direction and wind speed
df_wr = pd.read_csv("inputs/wind_rose.csv")

# Get the wind directions, wind speeds, and frequency table
wind_direction_values = df_wr["wd"].values
wind_speed_values = df_wr["ws"].values
wind_directions = df_wr["wd"].unique()
wind_speeds = df_wr["ws"].unique()
freq_vals = df_wr["freq_val"].values / df_wr["freq_val"].sum()

n_row = df_wr.shape[0]
n_wd = len(wind_directions)
n_ws = len(wind_speeds)

wd_step = wind_directions[1] - wind_directions[0]
ws_step = wind_speeds[1] - wind_speeds[0]

print("The wind rose dataframe looks as follows:")
print(df_wr.head())
print(f"There are {n_row} rows, {n_wd} unique wind directions, and {n_ws} unique wind speeds")
print(f"The wind direction has a step of {wd_step} and the wind speed has a step of {ws_step}")

# Declare a frequency table of size (n_wd, n_ws)
freq_table = np.zeros((n_wd, n_ws))

# Populate the frequency table using the values of wind_direction_values,
# wind_speed_values, and freq_vals
for i in range(n_row):
    wd = wind_direction_values[i]
    ws = wind_speed_values[i]
    freq = freq_vals[i]

    # Find the index of the wind direction and wind speed
    wd_idx = np.where(wind_directions == wd)[0][0]
    ws_idx = np.where(wind_speeds == ws)[0][0]

    # Populate the frequency table
    freq_table[wd_idx, ws_idx] = freq

# Normalize the frequency table
freq_table = freq_table / freq_table.sum()

print(f"The frequency table has shape {freq_table.shape}")

# Set up a wind rose
wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    freq_table=freq_table,
    ti_table=0.06,  # Assume contant TI
)

# Note that the wind rose could have been computed directly
# by first building a TimeSeries and applying
# the provided frequencies as bin weights in resampling
time_series = TimeSeries(
    wind_directions=wind_direction_values,
    wind_speeds=wind_speed_values,
    turbulence_intensities=0.06,
)

# Convert time series to wind rose using the frequencies as bin weights
wind_rose_from_time_series = time_series.to_wind_rose(
    wd_step=wd_step, ws_step=ws_step, bin_weights=freq_vals
)


print("Wind rose from wind_rose and wind_rose_from_time_series are equivalent:")
print(
    " -- Directions: "
    f"{np.allclose(wind_rose.wind_directions, wind_rose_from_time_series.wind_directions)}"
)
print(f" -- Speeds: {np.allclose(wind_rose.wind_speeds, wind_rose_from_time_series.wind_speeds)}")
print(f" -- Freq: {np.allclose(wind_rose.freq_table, wind_rose_from_time_series.freq_table)}")

# Set the wind rose
fmodel.set(wind_data=wind_rose)

# Note that the frequency table contains 0 frequency for some wind directions and wind speeds
# and we've not selected to compute 0 frequency bins, therefore the n_findex will be less than
# the total number of wind directions and wind speed combinations
print(f"Total number of rows in input wind rose: {n_row}")
print(f"n_findex: {fmodel.core.flow_field.n_findex}")

# Get the AEP
aep = fmodel.get_farm_AEP_with_wind_data(wind_rose)

# Print the AEP
print(f"AEP from wind rose: {aep/1E9:.1f} (GW-h)")
