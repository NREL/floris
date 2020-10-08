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


# Make in the inflow speed heterogenous

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import floris.tools as wfct


fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Set layout to 2 turbines
fi.reinitialize_flow_field(layout_array=[[0, 0], [100, 400]])

fi.calculate_wake()

# Get hor plane
hor_plane = fi.get_hor_plane()

# Introduce variation in wind speed
fi.reinitialize_flow_field(wind_speed=[6, 9], wind_layout=[[0, 0], [0, 500]])
fi.calculate_wake()
hor_plane_het_speed = fi.get_hor_plane()

# Plot
fig, axarr = plt.subplots(2, 1, figsize=(6, 10))

ax = axarr[0]
im = wfct.visualization.visualize_cut_plane(hor_plane, ax, minSpeed=4, maxSpeed=9)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Wind Speed (m/s)", labelpad=+10)
ax.set_title("Homogenous")

ax = axarr[1]
im = wfct.visualization.visualize_cut_plane(
    hor_plane_het_speed, ax, minSpeed=4, maxSpeed=9
)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Wind Speed (m/s)", labelpad=+10)
ax.set_title("Heterogenous")

plt.show()
