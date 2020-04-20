# Copyright 2020 NREL
 
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
import floris.tools as wfct
from floris.utilities import Vec3

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface('../example_input.json')

# Change the model to curl
fi.floris.farm.set_wake_model('curl')

# Change the layout
D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Calculate wake
fi.calculate_wake(yaw_angles=[25,0,0])

# Get the hor plane
hor_plane = fi.get_hor_plane()

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)

# Get the vertical cut through and visualize
cp = fi.get_cross_plane(5*D)
fig, ax = plt.subplots(figsize=(10,10))
wfct.visualization.visualize_cut_plane(cp, ax=ax,minSpeed=6.0,maxSpeed=8)
wfct.visualization.visualize_quiver(cp,ax=ax,downSamp=2)
ax.set_ylim([15,300])

# Save the flow data as vtk
flow_data = fi.get_flow_data()
flow_data = flow_data.crop(flow_data,[0,20*D],[-300,300],[50,300])
flow_data.save_as_vtk('for_3d_viz.vtk')

plt.show()
