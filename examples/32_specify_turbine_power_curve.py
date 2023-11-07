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

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
from floris.turbine_library.turbine_utilities import build_turbine_dict


"""
This example initializes the FLORIS software, and then uses internal
functions to run a simulation and plot the results. In this case,
we are plotting three slices of the resulting flow field:
1. Horizontal slice parallel to the ground and located at the hub height
2. Vertical slice of parallel with the direction of the wind
3. Veritical slice parallel to to the turbine disc plane

Additionally, an alternative method of plotting a horizontal slice
is shown. Rather than calculating points in the domain behind a turbine,
this method adds an additional turbine to the farm and moves it to
locations throughout the farm while calculating the velocity at it's
rotor.
"""

# Generate an example turbine power and thrust curve for use in the FLORIS 
# model
turbine_data_dict = {
    "wind_speed":[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "power_absolute":[0, 30, 200, 500, 1000, 2000, 4000, 4000, 4000, 4000, 4000],
    "thrust_coefficient":[0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
}

turbine_dict = build_turbine_dict(turbine_data_dict, "example_turbine")

fi = FlorisInterface("inputs/gch.yaml")
wind_speeds = np.linspace(1, 15, 100)
# Replace the turbine(s) in the FLORIS model with the created one
fi.reinitialize(layout_x=[0], layout_y=[0], wind_speeds=wind_speeds,
    turbine_type=[turbine_dict])
fi.calculate_wake()

powers = fi.get_farm_power()

fig, ax = plt.subplots(1,1)

ax.scatter(wind_speeds, powers[0,:]/1000, color="C0", s=5, label="Test points")
ax.scatter(turbine_data_dict["wind_speed"], turbine_data_dict["power_absolute"], 
    color="red", s=20, label="Specified points")

ax.grid()
ax.set_xlabel("Wind speed [m/s]")
ax.set_ylabel("Power [kW]")
ax.legend()

plt.show()