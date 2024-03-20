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

from floris import FlorisModel
import floris.flow_visualization as flowviz

"""
Example to test out using helix wake mixing of upstream turbines.
Helix wake mixing is turned on at turbine 1, off at turbines 2 to 4;
Turbine 2 is in wake turbine 1, turbine 4 in wake of turbine 3.
Will be refined before release.
TODO: merge the three examples into one.
"""

# Grab model of FLORIS and update to deratable turbines
fmodel = FlorisModel("inputs/emgauss_iea_15mw.yaml")
with open(str(
    fmodel.core.as_dict()["farm"]["turbine_library_path"] /
    (fmodel.core.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "helix"

# Set the wind directions and speeds to be constant over N different helix amplitudes
N = 1
helix_amplitudes = np.array([2.5, 0, 0, 0]).reshape(4, N).T

# Create 4 WT WF layout with lateral offset of 3D and streamwise offset of 4D
D = 240
fmodel.set(
    layout_x=[0.0, 4*D, 0.0, 4*D],
    layout_y=[0.0, 0.0, -3*D, -3*D],
    turbine_type=[turbine_type],
    wind_directions=270 * np.ones(N),
    wind_speeds=8.0 * np.ones(N),
    turbulence_intensities=0.06*np.ones(N),
    helix_amplitudes=helix_amplitudes
)
fmodel.run()
turbine_powers = fmodel.get_turbine_powers()

# Plot the flow fields for T1 helix_amplitude = 2.5
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=150.0,
    helix_amplitudes=helix_amplitudes
)

y_plane_baseline = fmodel.calculate_y_plane(
    x_resolution=200,
    z_resolution=100,
    crossstream_dist=0.0,
    helix_amplitudes=helix_amplitudes
)
y_plane_helix = fmodel.calculate_y_plane(
    x_resolution=200,
    z_resolution=100,
    crossstream_dist=-3*D,
    helix_amplitudes=helix_amplitudes
)

cross_plane = fmodel.calculate_cross_plane(
    y_resolution=100,
    z_resolution=100,
    downstream_dist=720.0,
    helix_amplitudes=helix_amplitudes
)

# Create the plots
fig, ax_list = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
ax_list = ax_list.flatten()
flowviz.visualize_cut_plane(
    horizontal_plane,
    ax=ax_list[0],
    label_contours=True,
    title="Horizontal"
)
flowviz.visualize_cut_plane(
    cross_plane,
    ax=ax_list[2],
    label_contours=True,
    title="Spanwise profile at 3D"
)

# fig2, ax_list2 = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
# ax_list2 = ax_list2.flatten()
flowviz.visualize_cut_plane(
    y_plane_baseline,
    ax=ax_list[1],
    label_contours=True,
    title="Streamwise profile, helix"
)
flowviz.visualize_cut_plane(
    y_plane_helix,
    ax=ax_list[3],
    label_contours=True,
    title="Streamwise profile, baseline"
)

# Calculate the effect of changing helix_amplitudes
N = 50
helix_amplitudes = np.array([
    np.linspace(0, 5, N),
    np.zeros(N), np.zeros(N), np.zeros(N)
    ]).reshape(4, N).T

# Reset FlorisModel for different helix amplitudes
fmodel.set(
    wind_directions=270 * np.ones(N),
    wind_speeds=8 * np.ones(N),
    turbulence_intensities=0.06*np.ones(N),
    helix_amplitudes=helix_amplitudes
    )
fmodel.run()
turbine_powers = fmodel.get_turbine_powers()

# Plot the power as a function of helix amplitude
fig_power, ax_power = plt.subplots(1, 1)
ax_power.plot(
    helix_amplitudes[:, 0],
    turbine_powers[:, 0]/1000,
    color="C0",
    label="Turbine 1"
)
ax_power.plot(
    helix_amplitudes[:, 0],
    turbine_powers[:, 1]/1000,
    color="C1",
    label="Turbine 2"
)
ax_power.plot(
    helix_amplitudes[:, 0],
    np.sum(turbine_powers[:,0:2], axis=1)/1000,
    color="C2",
    label="Turbines 1+2"
)
ax_power.plot(
    helix_amplitudes[:, 0],
    turbine_powers[:, 2]/1000,
    color="C0",
    linestyle="dotted", label="Turbine 3"
)
ax_power.plot(
    helix_amplitudes[:, 0],
    turbine_powers[:, 3]/1000,
    color="C1",
    linestyle="dotted", label="Turbine 4"
)
ax_power.plot(
    helix_amplitudes[:, 0],
    np.sum(turbine_powers[:, 2:], axis=1)/1000,
    color="C2",
    linestyle="dotted", label="Turbines 3+4"
)
ax_power.plot(
    helix_amplitudes[:, 0],
    np.ones(N)*np.max(turbine_type["power_thrust_table"]["power"]),
    color="k",
    linestyle="dashed", label="Rated power"
)
ax_power.grid()
ax_power.legend()
ax_power.set_xlim([0, 5])
ax_power.set_xlabel("Helix amplitude (deg)")
ax_power.set_ylabel("Power produced (kW)")
ax_power.set_title("Wind farm power production")

flowviz.show()
