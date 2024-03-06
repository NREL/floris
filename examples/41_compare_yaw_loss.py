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


"""
Test alternative models of loss to yawing
"""

# Parameters
N  = 101 # How many steps to cover yaw range in
yaw_max = 30 # Maximum yaw to test

# Set up the yaw angle sweep
yaw_angles = np.zeros((N,1))
yaw_angles[:,0] = np.linspace(-yaw_max, yaw_max, N)
print(yaw_angles.shape)



# Now loop over the operational models to compare
op_models = ["cosine-loss", "tum-loss"]
results = {}

for op_model in op_models:

    print(f"Evaluating model: {op_model}")

    # Grab model of FLORIS
    fi = FlorisInterface("inputs/gch.yaml")

    # Initialize to a simple 1 turbine case with n_findex = N
    fi.set(
        layout_x=[0],
        layout_y=[0],
        wind_directions=270 * np.ones(N),
        wind_speeds=8 * np.ones(N),
    )

    with open(str(
        fi.floris.as_dict()["farm"]["turbine_library_path"] /
        (fi.floris.as_dict()["farm"]["turbine_type"][0] + ".yaml")
    )) as t:
        turbine_type = yaml.safe_load(t)
    turbine_type["power_thrust_model"] = op_model

    # Change the turbine type
    fi.set(turbine_type=[turbine_type], yaw_angles=yaw_angles)

    # Calculate the power
    fi.run()
    turbine_power = fi.get_turbine_powers().squeeze()

    # Save the results
    results[op_model] = turbine_power

# Plot the results
fig, ax = plt.subplots()

colors = ["C0", "k", "r"]
linestyles = ["solid", "dashed", "dotted"]
for key, c, ls in zip(results, colors, linestyles):
    central_power = results[key][yaw_angles.squeeze() == 0]
    ax.plot(yaw_angles.squeeze(), results[key]/central_power, label=key, color=c, linestyle=ls)

ax.grid(True)
ax.legend()
ax.set_xlabel("Yaw angle [deg]")
ax.set_ylabel("Normalized turbine power [deg]")

plt.show()
