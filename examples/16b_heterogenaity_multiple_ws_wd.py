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
from floris.tools.visualization import visualize_cut_plane


"""
This example showcases the heterogeneous inflow capabilities of FLORIS
when multiple wind speeds and direction are considered.
"""


# Define the speed ups of the heterogeneous inflow, and their locations.
# For the 2-dimensional case, this requires x and y locations.
# The speed ups are multipliers of the ambient wind speed.
speed_ups = [[2.0, 1.0, 2.0, 1.0]]
x_locs = [-300.0, -300.0, 2600.0, 2600.0]
y_locs = [ -300.0, 300.0, -300.0, 300.0]

# Initialize FLORIS with the given input file via FlorisInterface.
# Note the heterogeneous inflow is defined in the input file.
fi = FlorisInterface("inputs/gch_heterogeneous_inflow.yaml")

# Set shear to 0.0 to highlight the heterogeneous inflow
fi.reinitialize(
    wind_shear=0.0,
    wind_speeds=[8.0],
    wind_directions=[270.],
    layout_x=[0, 0],
    layout_y=[-299., 299.],
)
fi.calculate_wake()
turbine_powers = fi.get_turbine_powers().flatten() / 1000.

# Show the initial results
print('------------------------------------------')
print('Given the speedups and turbine locations, ')
print(' the first turbine has an inflow wind speed')
print(' twice that of the second')
print(' Wind Speed = 8., Wind Direction = 270.')
print(f'T0: {turbine_powers[0]:.1f} kW')
print(f'T1: {turbine_powers[1]:.1f} kW')
print()

# Since het maps are assigned for each wind direciton, it's allowable to change
# the number of wind speeds
fi.reinitialize(wind_speeds=[4, 8])
fi.calculate_wake()
turbine_powers = np.round(fi.get_turbine_powers() / 1000.)
print('With wind speeds now set to 4 and 8 m/s')
print(f'T0: {turbine_powers[:, :, 0].flatten()} kW')
print(f'T1: {turbine_powers[:, :, 1].flatten()} kW')
print()

# To change the number of wind directions however it is necessary to make a matching
# change to the dimensions of the het map
speed_multipliers = [[2.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0]] # Expand to two wind directions
heterogenous_inflow_config = {
    'speed_multipliers': speed_multipliers,
    'x': x_locs,
    'y': y_locs,
}
fi.reinitialize(
    wind_directions=[270.0, 275.0],
    wind_speeds=[8.0],
    heterogenous_inflow_config=heterogenous_inflow_config
)
fi.calculate_wake()
turbine_powers = np.round(fi.get_turbine_powers() / 1000.)
print('With wind directions now set to 270 and 275 deg')
print(f'T0: {turbine_powers[:, :, 0].flatten()} kW')
print(f'T1: {turbine_powers[:, :, 1].flatten()} kW')

# # Uncomment if want to see example of error output
# # Note if we change wind directions to 3 without a matching change to het map we get an error
# print()
# print()
# print('~~ Now forcing an error by not matching wd and het_map')

# fi.reinitialize(wind_directions=[270, 275, 280], wind_speeds=[8.])
# fi.calculate_wake()
# turbine_powers = np.round(fi.get_turbine_powers() / 1000.)
