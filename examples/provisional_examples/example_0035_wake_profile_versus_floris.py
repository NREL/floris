#
# Copyright 2019 NREL
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

# See read the https://floris.readthedocs.io for documentation

import wfc_tools as wfct
import wfc_tools.visualization as vis
from wfc_tools.types import Vec3
import matplotlib.pyplot as plt
import numpy as np

# Floris setup
turbine1_x, turbine1_y = 500.0, 500.0
turbine2_x, turbine2_y = 1200, 500.0
floris_interface = wfct.floris_interface.FlorisInterface("example_input.json")

# Build the single turbine farm and run
floris_interface.floris.farm.set_turbine_locations([turbine1_x], [turbine1_y])
floris_interface.run_floris()
turbine1 = floris_interface.floris.farm.turbines[0]

# Get the FLORIS domain bounds and define a resolution
xmin, xmax, ymin, ymax, zmin, zmax = floris_interface.floris.farm.flow_field.domain_bounds
resolution = Vec3(
    1 + (xmax - xmin) / 10,
    1 + (ymax - ymin) / 10,
    1 + (zmax - zmin) / 10
)
floris_flow_field = floris_interface.get_flow_field(resolution=resolution)

# Visualization the flow field
fig, ax = plt.subplots()
hor_plane = wfct.cut_plane.HorPlane(floris_flow_field, turbine1.hub_height)
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
vis.plot_turbines(
    ax,
    floris_interface.floris.farm.layout_x,
    floris_interface.floris.farm.layout_y,
    floris_interface.get_yaw_angles(),
    floris_interface.floris.farm.turbine_map.turbines[0].rotor_diameter
)
ax.set_title('FLORIS')

# Use profile method to check power at 1200 m
floris_cross = wfct.cut_plane.CrossPlane(floris_flow_field, turbine2_x)

# What is the hypothetical power at this point?
hypothetical_power = wfct.cut_plane.calculate_power(
    floris_cross,
    x1_loc=turbine1_y,
    x2_loc=turbine1.hub_height,
    R=turbine1.rotor_radius,
    ws_array=turbine1.power_thrust_table["wind_speed"],
    cp_array=turbine1.power_thrust_table["power"]
)
print('Hypothetical Power = %.2f MW' % (hypothetical_power/1E6))

# What is the power from FLORIS================

# Add a second turbine
floris_interface.floris.farm.set_turbine_locations(
    [turbine1_x, turbine2_x],
    [turbine1_y, turbine1_y],
    calculate_wake=False
)
floris_interface.floris.farm.set_yaw_angles(0, calculate_wake=False)
floris_interface.run_floris()

# Visualize the two turbine case
floris_flow_field = floris_interface.get_flow_field(resolution=resolution)
fig, ax = plt.subplots()
hor_plane = wfct.cut_plane.HorPlane(floris_flow_field, turbine1.hub_height)
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
vis.plot_turbines(
    ax,
    floris_interface.floris.farm.layout_x,
    floris_interface.floris.farm.layout_y,
    floris_interface.get_yaw_angles(),
    turbine1.rotor_diameter
)
ax.set_title('FLORIS')

# Grab second turbine power
turbine2 = floris_interface.floris.farm.turbines[1]
print('FLORIS Power = %.2f MW' % (turbine2.power/1E6))

plt.show()
