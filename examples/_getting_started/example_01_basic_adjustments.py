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

# Load the input file as a dictionary so that we can modify it and pass to FlorisInterface
with open("../example_input.json") as json_file:
    base_input_dict = json.load(json_file)

fi = FlorisInterface(base_input_dict)

# Plot the initial setup
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[0], "Initial")

# Change the wind speed
base_input_dict["flow_field"]["wind_speeds"] = [7.5]
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[1], "WS=7")

# Change the wind direction
base_input_dict["flow_field"]["wind_directions"] = [320.0]
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[2], "WD=320.0")

# Change the TI
base_input_dict["flow_field"]["turbulence_intensity"] = 0.2
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[3], "TI=15%")

# Change the shear
base_input_dict["flow_field"]["wind_shear"] = 0.2
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[4], "Shear=.2")

# Change the veer
base_input_dict["flow_field"]["wind_veer"] = 5.0
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[5], "Veer=5")

# Change the air density
base_input_dict["flow_field"]["air_density"] = 1.0
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[6], "Air Density=1.0")

# Change the farm layout
base_input_dict["farm"]["layout_x"] = [0, 1000]
base_input_dict["farm"]["layout_y"] = [0, 0]
base_input_dict["farm"]["turbine_id"] = 2 * ["nrel_5mw"]
fi = FlorisInterface(base_input_dict)
fi.floris.solve_for_viz()
plot_slice_shortcut(fi, axarr[7], "Change layout")
plot_turbines_with_fi(axarr[7], fi)

# Changes the yaw angles
# fi = FlorisInterface(base_input_dict)
# fi.floris.farm.farm_controller.set_yaw_angles(np.array([25, 10, 0]))
# fi.floris.solve_for_viz()
# plot_slice_shortcut(fi, axarr[8], "Change yaw angles")
# plot_turbines_with_fi(axarr[8], fi)

plt.show()
