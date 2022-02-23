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
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from floris.tools import FlorisInterface

"""
This example demonstrates how to calculate the Annual Energy Production (AEP) of a wind farm using wind rose
information stored in a .csv file.

The wind rose information is first loaded, after which we initialize our Floris Interface. A 3 turbine farm is
generated, and then the turbine wakes and powers are calculated across all the wind directions. Finally, the farm power
is converted to AEP and reported out.
"""

# Read the windrose information file & normalize wind rose frequencies
fn = "inputs/wind_rose.csv"
df_wr = pd.read_csv(fn)

# Normalize the frequencies
df_wr["freq_val"] = df_wr["freq_val"].copy() / df_wr["freq_val"].sum()

# Split the wind rose into wind speeds above (df_wr_op)
# and below cut-in (df_wr_below_cut_in) as below cut-in winds are 0 power and
# 0 ambient wind speed generates warnings and nans in some of the wake models
df_wr_below_cut_in = df_wr[df_wr.ws < 3.0].copy()
df_wr_op = df_wr[df_wr.ws >= 3.0].copy()

# Derive the wind directions and speeds we need to evaluate in FLORIS
wd_array = np.array(df_wr_op["wd"].unique(), dtype=float)
ws_array = np.array(df_wr_op["ws"].unique(), dtype=float)

# Load the default example FLORIS object
fi = FlorisInterface("inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
# fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

# Assume a three-turbine wind farm with 5D spacing
D = 126.0 # Rotor diameter for the NREL 5 MW
fi.reinitialize(
    layout=[[0.0, 5* D, 10 * D], [0.0, 0.0, 0.0]],
    wind_directions=wd_array,
    wind_speeds=ws_array,
)

# Calculate FLORIS for every WD and WS combination
fi.calculate_wake()

# Return the farm power from the above calculation
farm_power_array = fi.get_farm_power()

# Now map the FLORIS solutions to the wind rose dataframe
wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing='ij')
interpolant = NearestNDInterpolator(
    np.vstack([wd_grid.flatten(), ws_grid.flatten()]).T, 
    farm_power_array.flatten()
)

# Use an interpolant to map the results back to the wind rose dataframe
# Technically this could be done directly but an interpolant is safer
# in the event the wind rose is irregular and/or ordered differently
# than floris
df_wr_op["farm_power"] = interpolant(df_wr_op[["wd", "ws"]])

# Recombine with the below cut-in data
df_wr_below_cut_in["farm_power"] = 0.
df_wr = df_wr_below_cut_in.append(df_wr_op).sort_values(["wd","ws"])

# Finally, calculate AEP in GWh
aep = np.dot(df_wr["freq_val"], df_wr["farm_power"]) * 365 * 24

print("Farm AEP: {:.3f} GWh".format(aep / 1.0e9))