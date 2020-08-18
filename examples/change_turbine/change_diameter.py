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


import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct


# Side by Side, adjust both T0 and T1 diameters
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.reinitialize_flow_field(layout_array=[[0, 0], [0, 1000]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])

for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    diameters = np.arange(80, 160, 1.0)
    powers = np.zeros_like(diameters)

    for d_idx, d in enumerate(diameters):
        fi.change_turbine([t], {"rotor_diameter": d})
        fi.calculate_wake()
        powers[d_idx] = fi.get_turbine_power()[t] / 1000.0

    ax.plot(diameters, powers, "k")
    ax.axhline(init_power[t], color="r", ls=":")
    ax.axvline(126, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([80, 160])
    ax.set_ylim([200, 3000])
    ax.set_xlabel("Diameter T%d" % t)
    ax.set_ylabel("Power")

plt.suptitle("Adjusting Both T0 and T1 Diameters")

# Waked, adjust T0 diameter
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])

for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    diameters = np.arange(80, 160, 1.0)
    powers = np.zeros_like(diameters)

    for d_idx, d in enumerate(diameters):
        fi.change_turbine([0], {"rotor_diameter": d})
        fi.calculate_wake()
        powers[d_idx] = fi.get_turbine_power()[t] / 1000.0

    ax.plot(diameters, powers, "k")
    ax.axhline(init_power[t], color="r", ls=":")
    ax.axvline(126, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([80, 160])
    ax.set_ylim([200, 3000])
    ax.set_xlabel("Diameter T0")
    ax.set_ylabel("Power")

plt.suptitle("Adjusting T0 Diameter")

# Waked, adjust T0 diameter
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
fi.reinitialize_flow_field(layout_array=[[0, 500], [0, 0]])

# Calculate wake
fi.calculate_wake()
init_power = np.array(fi.get_turbine_power()) / 1000.0

fig, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

# Show the hub-height slice in the 3rd pane
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[2])

for t in range(2):

    ax = axarr[t]

    # Now sweep the heights for this turbine
    diameters = np.arange(80, 160, 1.0)
    powers = np.zeros_like(diameters)

    for d_idx, d in enumerate(diameters):
        fi.change_turbine([1], {"rotor_diameter": d})
        fi.calculate_wake()
        powers[d_idx] = fi.get_turbine_power()[t] / 1000.0

    ax.plot(diameters, powers, "k")
    ax.axhline(init_power[t], color="r", ls=":")
    ax.axvline(126, color="r", ls=":")
    ax.set_title("T%d" % t)
    ax.set_xlim([80, 160])
    ax.set_ylim([200, 3000])
    ax.set_xlabel("Diameter T1")
    ax.set_ylabel("Power")

plt.suptitle("Adjusting T1 Diameter")

plt.show()
