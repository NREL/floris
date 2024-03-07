
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator

from floris.tools import FlorisInterface


"""
This example demonstrates how to calculate the Annual Energy Production (AEP)
of a wind farm using wind rose information stored in a .csv file.

The wind rose information is first loaded, after which we initialize our Floris
Interface. A 3 turbine farm is generated, and then the turbine wakes and powers
are calculated across all the wind directions. Finally, the farm power is
converted to AEP and reported out.
"""

# Read the windrose information file and display
df_wr = pd.read_csv("inputs/wind_rose.csv")
print("The wind rose dataframe looks as follows: \n\n {} \n".format(df_wr))

# Derive the wind directions and speeds we need to evaluate in FLORIS
wd_grid, ws_grid = np.meshgrid(
    np.array(df_wr["wd"].unique(), dtype=float),    # wind directions
    np.array(df_wr["ws"].unique(), dtype=float),    # wind speeds
    indexing="ij"
)
wind_directions = wd_grid.flatten()
wind_speeds = ws_grid.flatten()
turbulence_intensities = np.ones_like(wind_directions) * 0.06

# Format the frequency array into the conventional FLORIS v3 format, which is
# an np.array with shape (n_wind_directions, n_wind_speeds). To avoid having
# to manually derive how the variables are sorted and how to reshape the
# one-dimensional frequency array, we use a nearest neighbor interpolant. This
# ensures the frequency values are mapped appropriately to the new 2D array.
freq_interp = NearestNDInterpolator(df_wr[["wd", "ws"]], df_wr["freq_val"])
freq = freq_interp(wd_grid, ws_grid).flatten()

# Normalize the frequency array to sum to exactly 1.0
freq = freq / np.sum(freq)

# Load the FLORIS object
fi = FlorisInterface("inputs/gch.yaml") # GCH model
# fi = FlorisInterface("inputs/cc.yaml") # CumulativeCurl model

# Assume a three-turbine wind farm with 5D spacing. We reinitialize the
# floris object and assign the layout, wind speed and wind direction arrays.
D = fi.floris.farm.rotor_diameters[0] # Rotor diameter for the NREL 5 MW
fi.set(
    layout_x=[0.0, 5 * D, 10 * D],
    layout_y=[0.0, 0.0, 0.0],
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    turbulence_intensities=turbulence_intensities,
)

# Compute the AEP using the default settings
aep = fi.get_farm_AEP(freq=freq)
print("Farm AEP (default options): {:.3f} GWh".format(aep / 1.0e9))

# Compute the AEP again while specifying a cut-in and cut-out wind speed.
# The wake calculations are skipped for any wind speed below respectively
# above the cut-in and cut-out wind speed. This can speed up computation and
# prevent unexpected behavior for zero/negative and very high wind speeds.
# In this example, the results should not change between this and the default
# call to 'get_farm_AEP()'.
aep = fi.get_farm_AEP(
    freq=freq,
    cut_in_wind_speed=3.0,  # Wakes are not evaluated below this wind speed
    cut_out_wind_speed=25.0,  # Wakes are not evaluated above this wind speed
)
print("Farm AEP (with cut_in/out specified): {:.3f} GWh".format(aep / 1.0e9))

# Finally, we can also compute the AEP while ignoring all wake calculations.
# This can be useful to quantity the annual wake losses in the farm. Such
# calculations can be facilitated by enabling the 'no_wake' handle.
aep_no_wake = fi.get_farm_AEP(freq, no_wake=True)
print("Farm AEP (no_wake=True): {:.3f} GWh".format(aep_no_wake / 1.0e9))
