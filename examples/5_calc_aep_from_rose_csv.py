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

import os
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator
import floris.tools as wfct
from floris.tools import FlorisInterface


# Read the windrose information file & normalize wind rose frequencies
# root_path = os.path.dirname(os.path.abspath(__file__))
fn = "5_wind_rose.csv"
df_wr = pd.read_csv(fn)
df_wr["freq_val"] = df_wr["freq_val"] / df_wr["freq_val"].sum()
# Derive the wind directions and speeds we need to evaluate in floris
wd_array = np.array(df_wr["wd"].unique(), dtype=float)
ws_array = np.array(df_wr["ws"].unique(), dtype=float)
# Load the default example floris object
fi = FlorisInterface("inputs/gch.yaml")


# Assume a three-turbine wind farm with 5D spacing
fi.reinitialize(
    layout=[[0.0, 632.0, 1264.0], [0.0, 0.0, 0.0]],
    wind_directions=wd_array,
    wind_speeds=ws_array,
)

# Calculate FLORIS for every WD and WS combination
fi.calculate_wake()
farm_power_array = fi.get_farm_power()
# Now map the FLORIS solutions to the wind rose dataframe
X, Y = np.meshgrid(wd_array, ws_array, indexing='ij')
interpolant = NearestNDInterpolator(
    np.vstack([X.flatten(), Y.flatten()]).T, 
    farm_power_array.flatten()
)
df_wr["farm_power"] = interpolant(df_wr[["wd", "ws"]])
df_wr["farm_power"] = df_wr["farm_power"].fillna(0.0)
print("Farm solutions:")
print(df_wr)
# Finally, calculate AEP in GWh
aep = np.dot(df_wr["freq_val"], df_wr["farm_power"]) * 365 * 24
print("Farm AEP: {:.2f} GWh".format(aep / 1.0e9))