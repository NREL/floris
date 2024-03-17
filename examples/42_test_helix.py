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


"""
Example to test out using helix wake mixing of upstream turbines. 
Will be refined before release. 
TODO: merge the three examples into one.
"""

# Grab model of FLORIS and update to deratable turbines
fmodel = FlorisModel("inputs/emgauss_iea_15MW.yaml")
with open(str(
    fmodel.core.as_dict()["farm"]["turbine_library_path"] /
    (fmodel.core.as_dict()["farm"]["turbine_type"][0] + ".yaml")
)) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "helix"

# Set the wind directions and speeds to be constant over n_findex = N time steps
N = 50
fmodel.set(
    layout_x=[0, 2000.0],
    layout_y=[0.0, 0.0],
    turbine_type=[turbine_type],
    wind_directions=270 * np.ones(N),
    wind_speeds=10.0 * np.ones(N),
    turbulence_intensities=0.06*np.ones(N)
)
fmodel.run()
turbine_powers_orig = fmodel.get_turbine_powers()

# Add helix
helix_amplitudes = np.array([np.linspace(0, 5, N), np.zeros(N)]).reshape(2, N).T
fmodel.set(helix_amplitudes=helix_amplitudes)
fmodel.run()
turbine_powers_helix = fmodel.get_turbine_powers()

# Compute available power at downstream turbine
power_setpoints_2 = np.array([np.linspace(0, 0, N), np.full(N, None)]).T
fmodel.set(power_setpoints=power_setpoints_2)
fmodel.run()
turbine_powers_avail_ds = fmodel.get_turbine_powers()[:,1]

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

fig2, ax2 = plt.subplots()

ax2.plot(
    helix_amplitudes[:,0],
    turbine_powers_helix/turbine_powers_orig
)
ax2.plot(
    helix_amplitudes[:,0],
    np.sum(turbine_powers_helix, axis=1)/np.sum(turbine_powers_orig, axis=1)
)

ax2.legend(['Turbine 1','Turbine 2','Total power'])
ax2.grid()
ax2.set_title('Relative power gain, helix v baseline')
ax2.set_xlabel('Amplitude [deg]')
ax2.set_ylabel('Relative power [-]')

plt.show()

# fmodel.set()
