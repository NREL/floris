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


import numpy as np

from floris.tools import FlorisInterface


"""
This example follows the same setup as example 01 to createa a FLORIS instance and:
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations

with the modification of using a turbine definition that has a multi-dimensional Cp/Ct table.

In the input file `gch_multi_dim_cp_ct.yaml`, the turbine_type points to a turbine definition,
iea_15MW_floating_multi_dim_cp_ct.yaml located in the turbine_library,
that supplies a multi-dimensional Cp/Ct data file in the form of a .csv file. This .csv file
contains two additional conditions to define Cp/Ct values for: Tp for wave period, and Hs for wave
height. For every combination of Tp and Hs defined, a Cp/Ct/Wind speed table of values is also
defined. It is required for this .csv file to have the last 3 columns be ws, Cp, and Ct. In order
for this table to be used, the flag 'multi_dimensional_cp_ct' must be present and set to true in
the turbine definition. Also of note is the 'velocity_model' must be set to 'multidim_cp_ct' in
the main input file. With both of these values provided, the solver will downselect to use the
interpolant defined at the closest conditions. The user must supply these conditions in the
main input file under the 'flow_field' section, e.g.:

NOTE: The multi-dimensional Cp/Ct data used in this example is fictional for the purposes of
facilitating this example. The Cp/Ct values for the different wave conditions are scaled
values of the original Cp/Ct data for the IEA 15MW turbine.

flow_field:
  multidim_conditions:
    Tp: 2.5
    Hs: 3.01

The solver will then use the nearest-neighbor interpolant. These conditions are currently global
and used to select the interpolant at each turbine.

Also note in the example below that there is a specific method for computing powers when
using turbines with multi-dimensional Cp/Ct data under FlorisInterface, called
'get_turbine_powers_multidim'. The normal 'get_turbine_powers' method will not work.
"""

# Initialize FLORIS with the given input file via FlorisInterface.
fi = FlorisInterface("inputs/gch_multi_dim_cp_ct.yaml")

# Convert to a simple two turbine layout
fi.reinitialize(layout_x=[0., 500.], layout_y=[0., 0.])

# Single wind speed and wind direction
print('\n========================= Single Wind Direction and Wind Speed =========================')

# Get the turbine powers assuming 1 wind speed and 1 wind direction
fi.reinitialize(wind_directions=[270.0], wind_speeds=[8.0])

# Set the yaw angles to 0
yaw_angles = np.zeros([1, 2]) # 1 wind direction and wind speed, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)

# Get the turbine powers
turbine_powers = fi.get_turbine_powers_multidim() / 1000.0
print("The turbine power matrix should be of dimensions 1 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ",turbine_powers.shape)

# Single wind speed and multiple wind directions
print('\n========================= Single Wind Direction and Multiple Wind Speeds ===============')

wind_speeds = np.array([8.0, 9.0, 10.0])
wind_directions = np.array([270.0, 270.0, 270.0])

fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)
yaw_angles = np.zeros([3, 2])  # 3 wind directions/ speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers_multidim() / 1000.0
print("The turbine power matrix should be of dimensions 3 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ",turbine_powers.shape)

# Multiple wind speeds and multiple wind directions
print('\n========================= Multiple Wind Directions and Multiple Wind Speeds ============')

wind_speeds = np.tile([8.0, 9.0, 10.0], 3)
wind_directions = np.repeat([260.0, 270.0, 280.0], 3)

fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)
yaw_angles = np.zeros([9, 2])  # 9 wind directions/ speeds, 2 turbines
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers = fi.get_turbine_powers_multidim()/1000.
print("The turbine power matrix should be of dimensions 9 WD/WS X 2 Turbines")
print(turbine_powers)
print("Shape: ",turbine_powers.shape)
