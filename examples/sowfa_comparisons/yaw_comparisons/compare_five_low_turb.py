# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


# Compare 3 turbine results to SOWFA in 8 m/s, higher TI case

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import floris.tools as wfct


# HIGH TI

# Low TI

# Write out SOWFA results
layout_x = (1000.0, 1756.0, 2512.0, 3268.0, 4024.0)
layout_y = (1000.0, 1000.0, 1000.0, 1000.0, 1000.0)
sowfa_results = np.array(
    [
        [1946.3, 654.7, 764.8, 825, 819.8, 0, 0, 0, 0, 0],
        [1701.8, 947.9, 1091.7, 1037.9, 992.8, 20, 15, 10, 5, 0],
        [1587.2, 1202.3, 971.6, 857.3, 860.9, 25, 0, 0, 0, 0],
        [1588.4, 1007.8, 1207, 1190.9, 1173.2, 25, 20, 15, 10, 0],
        [1588.6, 928.6, 1428.6, 1031.1, 939.4, 25, 25, 0, 0, 0],
    ]
)
df_sowfa = pd.DataFrame(
    sowfa_results, columns=["p0", "p1", "p2", "p3", "p4", "y0", "y1", "y2", "y3", "y4"]
)

## SET UP FLORIS AND MATCH TO BASE CASE
wind_speed = 8.39
TI = 0.065

# Initialize the FLORIS interface fi, use default model
fi = wfct.floris_interface.FlorisInterface("../../example_input.json")
fi.reinitialize_flow_field(
    wind_speed=[wind_speed],
    turbulence_intensity=[TI],
    layout_array=(layout_x, layout_y),
)

# Setup alternative with gch off
fi_b = copy.deepcopy(fi)
fi_b.set_gch(False)

# Compare yaw combinations
yaw_combinations = [(0, 0, 0, 0, 0), (25, 0, 0, 0, 0), (25, 25, 0, 0, 0)]
yaw_names = ["%d/%d/%d/%d/%d" % yc for yc in yaw_combinations]

# Plot individual turbine powers
fig, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 5))

total_sowfa = []
total_gch_on = []
total_gch_off = []

for y_idx, yc in enumerate(yaw_combinations):

    # Collect SOWFA DATA
    s_data = df_sowfa[
        (df_sowfa.y0 == yc[0])
        & (df_sowfa.y1 == yc[1])
        & (df_sowfa.y2 == yc[2])
        & (df_sowfa.y2 == yc[3])
        & (df_sowfa.y2 == yc[4])
    ]
    s_data = [
        s_data.p0.values[0],
        s_data.p1.values[0],
        s_data.p2.values[0],
        s_data.p3.values[0],
        s_data.p4.values[0],
    ]
    total_sowfa.append(np.sum(s_data))

    # Collect GCH ON data
    fi.calculate_wake(yaw_angles=yc)
    g_data = np.array(fi.get_turbine_power()) / 1000.0
    total_gch_on.append(np.sum(g_data))

    # Collect GCH OFF data
    fi_b.calculate_wake(yaw_angles=yc)
    b_data = np.array(fi_b.get_turbine_power()) / 1000.0
    total_gch_off.append(np.sum(b_data))

    ax = axarr[y_idx]
    ax.set_title(yc)
    ax.plot(["T0", "T1", "T2", "T3", "T4"], s_data, "k", marker="s", label="SOWFA")
    ax.plot(["T0", "T1", "T2", "T3", "T4"], g_data, "g", marker="o", label="GCH ON")
    ax.plot(["T0", "T1", "T2", "T3", "T4"], b_data, "b", marker="*", label="GCH OFF")

axarr[-1].legend()

# Calculate totals and normalized totals
total_sowfa = np.array(total_sowfa)
nom_sowfa = total_sowfa / total_sowfa[0]

total_gch_on = np.array(total_gch_on)
nom_gch_on = total_gch_on / total_gch_on[0]

total_gch_off = np.array(total_gch_off)
nom_gch_off = total_gch_off / total_gch_off[0]

fig, axarr = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(8, 5))

# Show results
ax = axarr[0]
ax.set_title("Total Power")
ax.plot(yaw_names, total_sowfa, "k", marker="s", label="SOWFA", ls="None")
ax.axhline(total_sowfa[0], color="k", ls="--")
ax.plot(yaw_names, total_gch_on, "g", marker="o", label="GCH ON", ls="None")
ax.axhline(total_gch_on[0], color="g", ls="--")
ax.plot(yaw_names, total_gch_off, "b", marker="*", label="GCH OFF", ls="None")
ax.axhline(total_gch_off[0], color="b", ls="--")
ax.legend()

# Normalized results
ax = axarr[1]
ax.set_title("Normalized Power")
ax.plot(yaw_names, nom_sowfa, "k", marker="s", label="SOWFA", ls="None")
ax.axhline(nom_sowfa[0], color="k", ls="--")
ax.plot(yaw_names, nom_gch_on, "g", marker="o", label="GCH ON", ls="None")
ax.axhline(nom_gch_on[0], color="g", ls="--")
ax.plot(yaw_names, nom_gch_off, "b", marker="*", label="GCH OFF", ls="None")
ax.axhline(nom_gch_off[0], color="b", ls="--")

plt.show()
