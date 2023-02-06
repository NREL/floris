# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface, WindRose


"""
This example demonstrates how to calculate the Annual Energy Production (AEP)
of a wind farm using wind rose information stored in a .csv file.

The wind rose information is first loaded, after which we initialize our Floris
Interface. A 3 turbine farm is generated, and then the turbine wakes and powers
are calculated across all the wind directions. Finally, the farm power is
converted to AEP and reported out.
"""

# Read in the wind rose using the class
wind_rose = WindRose()
wind_rose.read_wind_rose_csv("inputs/wind_rose.csv")

# Show the wind rose
wind_rose.plot_wind_rose()

# Load the FLORIS object
fi = FlorisInterface("inputs/gch.yaml") # GCH model
# fi = FlorisInterface("inputs/cc.yaml") # CumulativeCurl model

# Assume a three-turbine wind farm with 5D spacing. We reinitialize the
# floris object and assign the layout, wind speed and wind direction arrays.
D = 126.0 # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout_x=[0.0, 5* D, 10 * D],
    layout_y=[0.0, 0.0, 0.0],
)

# Compute the AEP using the default settings
aep = fi.get_farm_AEP_wind_rose_class(wind_rose=wind_rose)
print("Farm AEP (default options): {:.3f} GWh".format(aep / 1.0e9))

# Compute the AEP again while specifying a cut-in and cut-out wind speed.
# The wake calculations are skipped for any wind speed below respectively
# above the cut-in and cut-out wind speed. This can speed up computation and
# prevent unexpected behavior for zero/negative and very high wind speeds.
# In this example, the results should not change between this and the default
# call to 'get_farm_AEP()'.
aep = fi.get_farm_AEP_wind_rose_class(
    wind_rose=wind_rose,
    cut_in_wind_speed=3.0,  # Wakes are not evaluated below this wind speed
    cut_out_wind_speed=25.0,  # Wakes are not evaluated above this wind speed
)
print("Farm AEP (with cut_in/out specified): {:.3f} GWh".format(aep / 1.0e9))

# Compute the AEP a final time, this time marking one of the turbines as
# belonging to another farm by setting its weight to 0
turbine_weights = np.array([1.0, 1.0, 0.0])
aep = fi.get_farm_AEP_wind_rose_class(
    wind_rose=wind_rose,
    turbine_weights= turbine_weights
)
print("Farm AEP (one turbine removed from power calculation): {:.3f} GWh".format(aep / 1.0e9))

# Finally, we can also compute the AEP while ignoring all wake calculations.
# This can be useful to quantity the annual wake losses in the farm. Such
# calculations can be facilitated by enabling the 'no_wake' handle.
aep_no_wake = fi.get_farm_AEP_wind_rose_class(wind_rose=wind_rose, no_wake=True)
print("Farm AEP (no_wake=True): {:.3f} GWh".format(aep_no_wake / 1.0e9))

wakeviz.show_plots()
