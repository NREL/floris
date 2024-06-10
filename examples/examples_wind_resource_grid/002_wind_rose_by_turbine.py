"""Example: Wind Rose By Turbine

An intended use case is to generate a WindRoseByTurbine object from the WindResourceGrid object.
The WindRoseByTurbine can be built by the WindRoseByTurbine by interpolating the Weibull parameters
to each turbine location specified by layout_x and layout_y and building a WindRose object for each
turbine.

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import WindResourceGrid, WindRoseByTurbine


# Read the WRG file
wrg = WindResourceGrid("wrg_example.wrg")

# Select a turbine layout
# The original grid in example 000 was defined by having the wind rose rotate with increasing y
# and move to higher speeds with increasing x.  Here we will define a turbine layout that
# has a line of turbine along the diagonals of the grid.
layout_x = np.array([0, 500, 1000])
layout_y = np.array([0, 1000, 2000])

# Get the WindRoseByTurbine object

wrt = wrg.get_wind_rose_by_turbine(layout_x, layout_y)

# Plot the wind roses
fig, axarr = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"polar": True})
wrt.plot_wind_roses(axarr=axarr, ws_step=5)
plt.show()
