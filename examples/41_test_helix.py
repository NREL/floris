# Copyright 2024 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import numpy as np
import yaml

from floris.tools import FlorisInterface
import floris.tools.visualization as wakeviz


"""
Example to test out derating of turbines and mixed derating and yawing. Will be refined before
release. TODO: Demonstrate shutting off turbines also, once developed.
"""

# Grab model of FLORIS and update to deratable turbines
fi = FlorisInterface("inputs/emgauss.yaml")
with open(str(
    fi.floris.as_dict()["farm"]["turbine_library_path"] /
    (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "helix"

# Convert to a simple two turbine layout with derating turbines
fi.reinitialize(layout_x=[0, 2000.0], layout_y=[0.0, 0.0], turbine_type=['iea_15mw'])

# Set the wind directions and speeds to be constant over n_findex = N time steps
N = 50
fi.reinitialize(wind_directions=270 * np.ones(N), wind_speeds=10.0 * np.ones(N))
fi.calculate_wake()
turbine_powers_orig = fi.get_turbine_powers()

# Add helix
helix_amplitudes = np.array([np.linspace(0, 5, N), np.zeros(N)]).reshape(2, N).T
fi.calculate_wake(helix_amplitudes=helix_amplitudes)
turbine_powers_helix = fi.get_turbine_powers()

# Compute available power at downstream turbine
power_setpoints_2 = np.array([np.linspace(0, 0, N), np.full(N, None)]).T
fi.calculate_wake(power_setpoints=power_setpoints_2)
turbine_powers_avail_ds = fi.get_turbine_powers()[:,1]

# Plot the results
fig, ax = plt.subplots(1, 1)
ax.plot(
    helix_amplitudes[:, 0], 
    turbine_powers_helix[:, 0]/1000, 
    color="C0", 
    label="Helix, turbine 1"
)
ax.plot(
    helix_amplitudes[:, 0],
    turbine_powers_helix[:, 1]/1000,
    color="C1",
    label="Helix, turbine 2"
)
ax.plot(
    helix_amplitudes[:, 0],
    np.sum(turbine_powers_helix, axis=1)/1000,
    color="C2",
    label="Helix, turbine 1+2"
)
ax.plot(
    helix_amplitudes[:, 0],
    turbine_powers_orig[:, 0]/1000,
    color="C0",
    linestyle="dotted", label="Helix, turbine 2"
)
ax.plot(
    helix_amplitudes[:, 0],
    turbine_powers_avail_ds/1000,
    color="C1",
    linestyle="dotted", label="Baseline, turbine 2"
)
ax.plot(
    helix_amplitudes[:, 0],
    np.sum(turbine_powers_orig, axis=1)/1000,
    color="C2",
    linestyle="dotted", label="Baseline, turbine 1+2"
)
ax.plot(
    helix_amplitudes[:, 0],
    np.ones(N)*np.max(turbine_type["power_thrust_table"]["power"]),
    color="k",
    linestyle="dashed", label="Rated power"
)
ax.grid()
ax.legend()
ax.set_xlim([0, 5])
ax.set_xlabel("Helix amplitude (deg)")
ax.set_ylabel("Power produced (kW)")

# Second example showing mixed model use.
# turbine_type["power_thrust_model"] = "mixed"
# yaw_angles = np.array([
#     [0.0, 0.0],
#     [0.0, 0.0],
#     [20.0, 10.0],
#     [0.0, 10.0],
#     [20.0, 0.0]
# ])
# power_setpoints = np.array([
#     [None, None],
#     [2e6, 1e6],
#     [None, None],
#     [2e6, None,],
#     [None, 1e6]
# ])
# fi.reinitialize(
#     wind_directions=270 * np.ones(len(yaw_angles)),
#     wind_speeds=10.0 * np.ones(len(yaw_angles)),
#     turbine_type=[turbine_type]*2
# )
# fi.calculate_wake(yaw_angles=yaw_angles, power_setpoints=power_setpoints)
# turbine_powers = fi.get_turbine_powers()
# print(turbine_powers)

# horizontal_plane = fi.calculate_horizontal_plane(
#     x_resolution=200,
#     y_resolution=100,
#     height=150.0,
#     yaw_angles=np.array([[25.,0.,0.]]),
# )

# fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
# ax_list = ax_list.flatten()
# wakeviz.visualize_cut_plane(
#     horizontal_plane,
#     ax=ax_list[0],
#     label_contours=True,
#     title="Horizontal"
# )

plt.show()
