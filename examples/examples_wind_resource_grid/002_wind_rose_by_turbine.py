"""Example: Wind Rose By Turbine

An intended use case is to generate a WindRoseByTurbine object from the WindResourceGrid object.
The WindRoseByTurbine can be built by the WindRoseByTurbine by interpolating the Weibull parameters
to each turbine location specified by layout_x and layout_y and building a WindRose object for each
turbine.

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    WindResourceGrid,
    WindRoseByTurbine,
)


# Read the WRG file
wrg = WindResourceGrid("wrg_example.wrg")

# Select a turbine layout
# The original grid in example 000 was defined by having the wind rose rotate with increasing y
# and move to higher speeds with increasing x.  Here we will define a turbine layout that
# has a line of turbine along the diagonals of the grid.
layout_x = np.array([0, 500, 1000])
layout_y = np.array([0, 1000, 2000])

# Get the WindRoseByTurbine object
wind_rose_by_turbine = wrg.get_wind_rose_by_turbine(layout_x, layout_y)

# Plot the wind roses in the first figure
fig, axarr = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"polar": True})
wind_rose_by_turbine.plot_wind_roses(axarr=axarr, ws_step=5)

# Now apply the wind_rose_by_turbine to a FlorisModel
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_by_turbine)
fmodel.run()
expected_turbine_powers = fmodel.get_expected_turbine_powers()

# Compare with the result if just using the first wind rose or the last wind rose
fmodel.set(wind_data=wind_rose_by_turbine.wind_roses[0])
fmodel.run()
expected_turbine_powers_first = fmodel.get_expected_turbine_powers()

fmodel.set(wind_data=wind_rose_by_turbine.wind_roses[-1])
fmodel.run()
expected_turbine_powers_last = fmodel.get_expected_turbine_powers()

# Print the results
print("Expected turbine powers:")
print("All wind roses:", expected_turbine_powers)
print("First wind rose:", expected_turbine_powers_first)
print("Last wind rose:", expected_turbine_powers_last)

# Compare expected farm power against the sum of expected_turbine_power
expected_farm_power = fmodel.get_expected_farm_power()
print("Expected farm power:", expected_farm_power)
print("Sum of expected turbine powers:", expected_turbine_powers_last.sum())

plt.show()
