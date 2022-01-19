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


import json
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface
from floris.tools import visualize_cut_plane, plot_turbines_with_fi


"""
This example reviews two essential functions of the FLORIS interface
reinitialize_flow_field and calculate_wake

reinitialize_flow_field is used to change the layout and inflow of the farm
while calculate_wake computed the wake velocities, deflections and combinations

Both functions provide a simpler interface to the underlying functions in the FLORIS class

Using them ensures that necessary recalcuations occur with changing certain variables

Note that it is typically necessary to call calculate_wake after reinitialize_flow_field,
but the two functions are seperated so that calculate_wake can be called repeatedly,
for example when optimizing yaw angles
"""

# Declare a short-cut visualization function for brevity in this example
def plot_slice_shortcut(fi, ax, title):
    # Get horizontal plane at default height (hub-height)
    hor_plane = fi.get_hor_plane()
    visualize_cut_plane(hor_plane, ax=ax, minSpeed=4.0, maxSpeed=8.0)


# Define a plot
fig, axarr = plt.subplots(3, 3, sharex=True, figsize=(12, 5))
axarr = axarr.flatten()


fi = FlorisInterface("inputs/input_gch.yaml")
solver_settings = {
    "type": "flow_field_grid",
    "flow_field_grid_points": [200,100,7]
}
fi.reinitialize(solver_settings=solver_settings)


# Plot the initial setup
horizontal_plane = fi.get_hor_plane()
visualize_cut_plane(horizontal_plane, ax=axarr[0], minSpeed=4.0, maxSpeed=8.0)


# Change the wind speed
horizontal_plane = fi.get_hor_plane(ws=[7.0])
visualize_cut_plane(horizontal_plane, ax=axarr[1], minSpeed=4.0, maxSpeed=8.0)


# # Change the wind direction
horizontal_plane = fi.get_hor_plane(wd=[320.0], ws=[8.0])
visualize_cut_plane(horizontal_plane, ax=axarr[2], minSpeed=4.0, maxSpeed=8.0)


# # Change the TI
# fi.reinitialize(turbulence_intensity=0.2)
# fi.floris.solve_for_viz()
# plot_slice_shortcut(fi, axarr[3], "TI=15%")


# # Change the shear
# fi.reinitialize(wind_shear=0.2)
# fi.floris.solve_for_viz()
# plot_slice_shortcut(fi, axarr[4], "Shear=.2")


# # Change the veer
# fi.reinitialize(wind_veer=5.0)
# fi.floris.solve_for_viz()
# plot_slice_shortcut(fi, axarr[5], "Veer=5")


# Change the farm layout
N = 3  # Number of turbines per row and per column
X, Y = np.meshgrid(
    5.0 * fi.floris.turbine.rotor_diameter * np.arange(0, N, 1),
    5.0 * fi.floris.turbine.rotor_diameter * np.arange(0, N, 1),
)
fi.reinitialize( layout=( X.flatten(), Y.flatten() ) )
horizontal_plane = fi.get_hor_plane()
visualize_cut_plane(horizontal_plane, ax=axarr[7], minSpeed=4.0, maxSpeed=8.0)
# plot_turbines_with_fi(axarr[7], fi)


# Change the yaw angles
yaw_angles = np.zeros((1, 1, N*N))
yaw_angles[:,:,0] = 20.0  # Yaw the leading turbine 20 degrees
horizontal_plane = fi.get_hor_plane(yaw_angles=yaw_angles)
visualize_cut_plane(horizontal_plane, ax=axarr[8], minSpeed=4.0, maxSpeed=8.0)
# plot_turbines_with_fi(axarr[8], fi)

plt.show()
