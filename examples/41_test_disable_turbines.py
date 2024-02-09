# Copyright 2023 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# Example adapted from https://github.com/NREL/floris/pull/693 contributed by Elie Kadoche


import matplotlib.pyplot as plt
import numpy as np
import yaml

from floris.tools import FlorisInterface


"""
This example demonstrates the ability of FLORIS to shut down some turbines
during a simulation.
"""

# Initialize the FLORIS interface
fi = FlorisInterface("inputs/gch.yaml")

# Change to the mixed model turbine
with open(
    str(
        fi.floris.as_dict()["farm"]["turbine_library_path"]
        / (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
    )
) as t:
    turbine_type = yaml.safe_load(t)
turbine_type["power_thrust_model"] = "mixed"
fi.reinitialize(turbine_type=[turbine_type])

# Consider a wind farm of 3 aligned wind turbines
layout = np.array([[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0]])

# Run the computations for 2 identical wind data
# (n_findex = 2)
wind_directions = np.array([270.0, 270.0])
wind_speeds = np.array([8.0, 8.0])

# Shut down the first 2 turbines for the second findex
# 2 findex x 3 turbines
disable_turbines = np.array([[False, False, False], [True, True, False]])

# Simulation
# ------------------------------------------

# Reinitialize flow field
fi.reinitialize(
    layout_x=layout[:, 0],
    layout_y=layout[:, 1],
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
)

# # Compute wakes
fi.calculate_wake(disable_turbines=disable_turbines)

# Results
# ------------------------------------------

# Get powers and effective wind speeds
turbine_powers = fi.get_turbine_powers()
turbine_powers = np.round(turbine_powers * 1e-3, decimals=2)
effective_wind_speeds = fi.turbine_average_velocities


# Plot the results
fig, axarr = plt.subplots(2, 1, sharex=True)

# Plot the power
ax = axarr[0]
ax.plot(["T0", "T1", "T2"], turbine_powers[0, :], "ks-", label="All On")
ax.plot(["T0", "T1", "T2"], turbine_powers[1, :], "ro-", label="T0/T1 Disabled")
ax.set_ylabel("Power (kW)")
ax.grid(True)
ax.legend()

ax = axarr[1]
ax.plot(["T0", "T1", "T2"], effective_wind_speeds[0, :], "ks-", label="All On")
ax.plot(["T0", "T1", "T2"], effective_wind_speeds[1, :], "ro-", label="T0/T1 Disabled")
ax.set_ylabel("Effective Wind Speeds (m/s)")
ax.grid(True)
ax.legend()

plt.show()
