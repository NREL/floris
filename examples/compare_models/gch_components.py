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


# Compare 5 turbine results to SOWFA in 8 m/s, higher TI case

import copy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import floris.tools as wfct


# Set up default model
fi_gch = wfct.floris_interface.FlorisInterface("../example_input.json")

# Set as a row of 5 turbines spaced 6D apart
D = 126.0
fi_gch.reinitialize_flow_field(
    layout_array=[[D * 6 * r for r in range(5)], [0, 0, 0, 0, 0]]
)

# Set up versions with components disabled
fi_ss = copy.deepcopy(fi_gch)
fi_ss.set_gch_yaw_added_recovery(False)

# Set up versions with components disabled
fi_yar = copy.deepcopy(fi_gch)
fi_yar.set_gch_secondary_steering(False)

# Set up versions with components disabled
fi_all_off = copy.deepcopy(fi_gch)
fi_all_off.set_gch(False)

# Get the baseline power for all allgined
fi_gch.calculate_wake()
base_power = np.array(fi_gch.get_turbine_power()) / 1000.0

fi_gch.calculate_wake([30, 0, 0, 0, 0])
gch_power = np.array(fi_gch.get_turbine_power()) / 1000.0

fi_ss.calculate_wake([30, 0, 0, 0, 0])
ss_power = np.array(fi_ss.get_turbine_power()) / 1000.0

fi_yar.calculate_wake([30, 0, 0, 0, 0])
yar_power = np.array(fi_yar.get_turbine_power()) / 1000.0

fi_all_off.calculate_wake([30, 0, 0, 0, 0])
all_off_power = np.array(fi_all_off.get_turbine_power()) / 1000.0

turbines = ["T1", "T2", "T3", "T4", "T5"]
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(turbines, base_power, "k.-", label="Aligned")
ax.plot(turbines, all_off_power, "bd-", label="T1 yawed, SS OFF, YAR OFF")
ax.plot(turbines, yar_power, "y^-", label="T1 yawed, SS OFF, YAR ON")
ax.plot(turbines, ss_power, "m*-", label="T1 yawed, SS ON, YAR OFF")
ax.plot(turbines, gch_power, "gs-", label="T1 yawed, SS ON, YAR ON")
ax.grid(True)
ax.set_xlabel("Turbinbe")
ax.set_ylabel("Power (kW)")
ax.legend()

# Total power
print("Total Powers =======")
print("Baseline:\t%.1f" % np.sum(base_power))
print("All Off:\t%.1f" % np.sum(all_off_power))
print("YAR:\t%.1f" % np.sum(yar_power))
print("SS:\t%.1f" % np.sum(ss_power))
print("GCH:\t%.1f" % np.sum(gch_power))

plt.show()
