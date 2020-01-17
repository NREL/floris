# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import pandas as pd
import numpy as np

print('-------- Initializing FLORIS ------')

fi = wfct.floris_interface.FlorisInterface("heterogeneous_input.json")
rotor_diameter = [80]*7
fi.set_rotor_diameter(rotor_diameter = rotor_diameter)

print('------------- Visualizing initial cut plane --------')

hor_plane = fi.get_hor_plane()

wind_direction_at_turbine = fi.floris.farm.wind_map.turbine_wind_direction
fig, ax=plt.subplots(figsize = (10,7))
im = wfct.visualization.visualize_cut_plane(hor_plane, ax)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Wind Speed (m/s)", labelpad=+10)
im = wfct.visualization.plot_turbines(ax = ax, layout_x = fi.layout_x ,layout_y = fi.layout_y, yaw_angles = [-1 * d for i,d in enumerate(wind_direction_at_turbine)], D = 80 )
plt.show(im)


# Reinitialize Flow Field with new measurements
speed=[6,8,6,10]
ti = [0.05, 0.1, 0.04, 0.3]
wdir=[250, 330, 250, 330]


print('--------- reinitializing flow field ----------')
fi.reinitialize_flow_field(wind_speed=speed, 
                           wind_direction = wdir, 
                           turbulence_intensity = ti
                          )
fi.calculate_wake()

print('------------- visualizing reinitialized cut plane --------')

hor_plane = fi.get_hor_plane()


wind_direction_at_turbine = fi.floris.farm.wind_map.turbine_wind_direction
           

fig, ax=plt.subplots(figsize = (10,7))
im = wfct.visualization.visualize_cut_plane(hor_plane, ax)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Wind Speed (m/s)", labelpad=+10)
im = wfct.visualization.plot_turbines(ax = ax, layout_x = fi.layout_x ,layout_y = fi.layout_y, yaw_angles =  [-1 * d for i,d in enumerate(wind_direction_at_turbine)], D = 80 )
plt.show(im)



# Visualize Percent Velocity Deficit
print('------------- visualizing percent velocity deficit in cut plane --------')

hor_plane = fi.get_hor_plane()


wind_direction_at_turbine = fi.floris.farm.wind_map.turbine_wind_direction

fig, ax=plt.subplots(figsize = (10,7))
im = wfct.visualization.visualize_cut_plane(hor_plane, ax)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Velocity Deficit (%)", labelpad=+10)
im = wfct.visualization.plot_turbines(ax = ax, layout_x = fi.layout_x ,layout_y = fi.layout_y, yaw_angles =  [-1 * d for i,d in enumerate(wind_direction_at_turbine)], D = 80 )
plt.show(im)

# reinitialize flow field using one measurement
layout_x = [0, 630, 1260] 
layout_y = [0, 0, 0] 
wind_x = [0]
wind_y = [0]
speed=[8]
ti = [0.1]
wdir=[270]


print('--------- reinitialize homogeneous flow field with new layout ----------')
fi.reinitialize_flow_field(wind_speed=speed, 
                           wind_direction = wdir, 
                           turbulence_intensity = ti,
                           wind_layout = (wind_x,wind_y),
                           layout_array = (layout_x,layout_y))
print('------------- visualizing reinitialized cut plane --------')
fi.calculate_wake()
hor_plane = fi.get_hor_plane()


wind_direction_at_turbine = fi.floris.farm.wind_map.turbine_wind_direction

fig, ax=plt.subplots(figsize = (11,3))
im = wfct.visualization.visualize_cut_plane(hor_plane, ax)
cbar = fig.colorbar(im, ax=ax, fraction = 0.025, aspect = 12,  pad=0.04)
cbar.set_label("Wind Speed (m/s)", labelpad=+10)
im = wfct.visualization.plot_turbines(ax = ax, layout_x = fi.layout_x ,layout_y = fi.layout_y, yaw_angles =  [-1 * d for i,d in enumerate(wind_direction_at_turbine)], D = 80 )
plt.show(im)
