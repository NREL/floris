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


import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


# Instantiate FLORIS using either the GCH or CC model
fi = FlorisInterface("/Users/rmudafor/Development/floris/examples/inputs/jensen.yaml")
input_dict = fi.floris.as_dict()
input_dict["solver"] = {"type": "cubature_grid"}
fi = FlorisInterface(input_dict)

wd_array = np.arange(250.0, 251.0, 1.0)
fi.reinitialize(wind_directions=wd_array)
fi.calculate_wake()

# Collect the turbine powers
turbine_powers = fi.get_turbine_powers() / 1e3  # In kW

# Pull out the power values per turbine
pow_t0 = turbine_powers[:, :, 0].flatten()
pow_t1 = turbine_powers[:, :, 1].flatten()

# Plot turbine interpolation scheme for first WTG
fig, ax = plt.subplots(nrows=2)
ti = 0
hh = fi.floris.farm.turbine_definitions[ti]["hub_height"]
r = fi.floris.farm.turbine_definitions[ti]["rotor_diameter"] / 2.0
theta = np.linspace(0.0, 2 * np.pi, 1000)
sc = ax[0].scatter(
    fi.floris.grid.y[0, 0, ti].flatten(),
    fi.floris.grid.z[0, 0, ti].flatten(),
    # c=fi.floris.grid.weights[0, 0, ti, :, :].flatten()
)
# plt.colorbar(sc, ax=ax[0])
ax[0].plot(fi.floris.grid.y[0, 0, ti, :, :].mean() + r * np.cos(theta), hh + r * np.sin(theta), '--')
ax[0].grid(True)
ax[0].axis("equal")
ax[0].set_title("Rotor interpolation points")
ax[0].set_xlabel(" y (m)")
ax[0].set_ylabel(" z (m)")

# Plot wakes
ax[1].plot(wd_array, pow_t0, "-o", color="k", label="Upstream Turbine")
ax[1].plot(wd_array, pow_t1, "-o", color="r", label="Downstream Turbine")
ax[1].grid(True)
ax[1].legend()
ax[1].set_xlabel("Wind Direction (deg)")
ax[1].set_ylabel("Power (kW)")
ax[1].set_title("Wake loss profile")

plt.show()
