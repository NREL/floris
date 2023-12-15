# Copyright 2021 NREL

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
from floris.turbine_library.turbine_utilities import build_turbine_dict


"""
This example demonstrates how to specify a turbine model based on a power
and thrust curve for the wind turbine, as well as possible physical parameters
(which default to the parameters of the NREL 5MW reference turbine).

Note that it is also possible to have a .yaml created, if the file_path
argument to build_turbine_dict is set.
"""

# Generate an example turbine power and thrust curve for use in the FLORIS model
turbine_data_dict = {
    "wind_speed":[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "power_absolute":[0, 30, 200, 500, 1000, 2000, 4000, 4000, 4000, 4000, 4000],
    "thrust_coefficient":[0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
}

turbine_dict = build_turbine_dict(
    turbine_data_dict,
    "example_turbine",
    file_path=None,
    generator_efficiency=1,
    hub_height=90,
    pP=1.88,
    pT=1.88,
    rotor_diameter=126,
    TSR=8,
    air_density=1.225,
    ref_tilt_cp_ct=5
)

fi = FlorisInterface("inputs/gch.yaml")
wind_speeds = np.linspace(1, 15, 100)
wind_directions = 270 * np.ones_like(wind_speeds)
# Replace the turbine(s) in the FLORIS model with the created one
fi.reinitialize(
    layout_x=[0],
    layout_y=[0],
    wind_speeds=wind_speeds,
    wind_directions=wind_directions,
    turbine_type=[turbine_dict]
)
fi.calculate_wake()

powers = fi.get_farm_power()

fig, ax = plt.subplots(1,1)

ax.scatter(wind_speeds, powers/1000, color="C0", s=5, label="Test points")
ax.scatter(turbine_data_dict["wind_speed"], turbine_data_dict["power_absolute"],
    color="red", s=20, label="Specified points")

ax.grid()
ax.set_xlabel("Wind speed [m/s]")
ax.set_ylabel("Power [kW]")
ax.legend()

plt.show()
