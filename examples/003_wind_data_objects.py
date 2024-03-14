"""Example 3: Wind Data Objects

This example demonstrates the use of wind data objects in FLORIS:
 TimeSeries,
 WindRose, and WindTIRose.


Main concept is introduce FLORIS and illustrate essential structure
of most-used FLORIS calls
"""


import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
    WindTIRose,
)


##################################################
# Initializing
##################################################

# FLORIS provides a set of wind data objects to hold the ambient wind conditions in a
# convenient classes that include capabilities and methods to manipulate and visualize
# the data.

# The TimeSeries class is used to hold time series data, such as wind speed, wind direction,
# and turbulence intensity.

# Generate wind speeds, directions, and turbulence intensities via random signals
N = 100
wind_speeds = 8 + 2 * np.random.randn(N)
wind_directions = 270 + 30 * np.random.randn(N)
turbulence_intensities = 0.06 + 0.02 * np.random.randn(N)

time_series = TimeSeries(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    turbulence_intensities=turbulence_intensities,
)

# The WindRose class is used to hold wind rose data, such as wind speed, wind direction,
# and frequency.  TI is represented as a bin average per wind direction and speed bin.
wind_directions = np.arange(0, 360, 3.0)
wind_speeds = np.arange(4, 20, 2.0)

# Make TI table 6% TI for all wind directions and speeds
ti_table = 0.06 * np.ones((len(wind_directions), len(wind_speeds)))

# Uniform frequency
freq_table = np.ones((len(wind_directions), len(wind_speeds)))
freq_table = freq_table / np.sum(freq_table)

wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    ti_table=ti_table,
    freq_table=freq_table,
)

# The WindTIRose class is similar to the WindRose table except that TI is also binned
# making the frequency table a 3D array.
turbulence_intensities = np.arange(0.05, 0.15, 0.01)

# Uniform frequency
freq_table = np.ones((len(wind_directions), len(wind_speeds), len(turbulence_intensities)))

wind_ti_rose = WindTIRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    turbulence_intensities=turbulence_intensities,
    freq_table=freq_table,
)

##################################################
# Broadcasting
##################################################

# A convenience method of the wind data objects is that, unlike the lower-level
# FlorisModel.set() method, the wind data objects can broadcast upward data provided
# as a scalar to the full array.  This is useful for setting the same wind conditions
# for all turbines in a wind farm.

# For TimeSeries, as long as one condition is given as an array, the other 2
# conditions can be given as scalars.  The TimeSeries object will broadcast the
# scalars to the full array (uniform)
wind_directions = 270 + 30 * np.random.randn(N)
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=8.0, turbulence_intensities=0.06
)


# For WindRose, wind directions and wind speeds must be given as arrays, but the
# ti_table can be supplied as a scalar which will apply uniformly to all wind
# directions and speeds.  Not supplying a freq table will similarly generate
# a uniform frequency table.
wind_directions = np.arange(0, 360, 3.0)
wind_speeds = np.arange(4, 20, 2.0)
wind_rose = WindRose(wind_directions=wind_directions, wind_speeds=wind_speeds, ti_table=0.06)


##################################################
# Wind Rose from Time Series
##################################################

# The TimeSeries class has a method to generate a wind rose from a time series based on binning
wind_rose = time_series.to_wind_rose(
    wd_edges=np.arange(0, 360, 3.0), ws_edges=np.arange(4, 20, 2.0)
)


##################################################
# Setting turbulence intensity
##################################################

# Each of the wind data objects also has the ability to set the turbulence intensity
# according to a function of wind speed and direction.  This can be done using
# a custom function by using the function assign_ti_using_IEC_method which assigns
# TI based on the IEC 61400-1 standard
wind_rose.assign_ti_using_IEC_method()  # Assign using default settings for Iref and offset


##################################################
# Plotting Wind Data Objects
##################################################

# Certain plotting methods are included to enable visualization of the wind data objects
# Plotting a wind rose
wind_rose.plot_wind_rose()

# Showing TI over wind speed for a WindRose
wind_rose.plot_ti_over_ws()

##################################################
# Assigning value to wind data objects
##################################################

# Wind data objects can also hold value information, such as the price of electricity for different
# time periods or wind conditions.  These can then be used in later optimization methods to optimize
# for quantities besides AEP.

N = 100
wind_speeds = 8 + 2 * np.random.randn(N)
values = 1 / wind_speeds  # Assume Value is inversely proportional to wind speed

time_series = TimeSeries(
    wind_directions=270.0, wind_speeds=wind_speeds, turbulence_intensities=0.06, values=values
)

##################################################
# Setting the FLORIS model via wind data
##################################################

# Each of the wind data objects can be used to set the FLORIS model by passing
# them in as is to the set method.  The FLORIS model will then use the member functions
# of the wind data to extract the wind conditions for the simulation.  Frequency tables
# are also extracted for AEP calculations.

fmodel = FlorisModel("inputs/gch.yaml")

# Set the wind conditions using the TimeSeries object
fmodel.set(wind_data=time_series)

# Set the wind conditions using the WindRose object
fmodel.set(wind_data=wind_rose)

# Note that in the case of the wind_rose, under the default settings, wind direction and wind speed
# bins for which frequency is zero are not simulated.  This can be changed by setting the
# compute_zero_freq_occurrence parameter to True.
wind_directions = np.array([200.0, 300.0])
wind_speeds = np.array([5.0, 1.00])
freq_table = np.array([[0.5, 0], [0.5, 0]])
wind_rose = WindRose(
    wind_directions=wind_directions, wind_speeds=wind_speeds, ti_table=0.06, freq_table=freq_table
)
fmodel.set(wind_data=wind_rose)

print(
    f"Number of conditions to simulate with compute_zero_freq_occurrence"
    f"False: {fmodel.core.flow_field.n_findex}"
)

wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    ti_table=0.06,
    freq_table=freq_table,
    compute_zero_freq_occurrence=True,
)
fmodel.set(wind_data=wind_rose)

print(
    f"Number of conditions to simulate with compute_zero_freq_occurrence"
    f"True: {fmodel.core.flow_field.n_findex}"
)

# Set the wind conditions using the WindTIRose object
fmodel.set(wind_data=wind_ti_rose)

plt.show()
