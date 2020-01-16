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

import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
import matplotlib.pyplot as plt
import numpy as np

# Define a minspeed and maxspeed to use across visualiztions
minspeed = 4.0
maxspeed = 8.5

# Load the SOWFA case in
si = wfct.sowfa_utilities.SowfaInterface('sowfa_example')
sowfa_flow_data = si.flow_data

# Load the FLORIS case in
fi = wfct.floris_interface.FlorisInterface("example_input.json")
fi.calculate_wake()

# Set the relevant FLORIS parameters to equal the SOWFA case
fi.reinitialize_flow_field(wind_speed=[si.precursor_wind_speed],
                           wind_direction=[si.precursor_wind_dir],
                           layout_array=(si.layout_x, si.layout_y)
                           )

# Set the yaw angles
fi.calculate_wake(yaw_angles=si.yaw_angles)

# Show projected and unprojected cut planes
x_loc = 600

cut_plane_sowfa = si.get_cross_plane(x_loc)
cut_plane_floris = fi.get_cross_plane(x_loc)
cut_plane_floris_project = cp.project_onto(cut_plane_floris,cut_plane_sowfa)
cut_plane_difference = cp.subtract(cut_plane_sowfa,cut_plane_floris_project)

print(cut_plane_sowfa.df.head())
print(cut_plane_floris_project.df.head())
print(cut_plane_difference.df.head())

fig, axarr = plt.subplots(2,2,figsize=(10,10))

# SOWFA
ax = axarr[0,0]
wfct.visualization.visualize_cut_plane(
     cut_plane_sowfa, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
ax.set_title('SOWFA')

# FLORIS
ax = axarr[0,1]
wfct.visualization.visualize_cut_plane(
     cut_plane_floris, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
ax.set_title('FLORIS')

# FLORIS Project
ax = axarr[1,0]
wfct.visualization.visualize_cut_plane(
     cut_plane_floris_project, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
ax.set_title('FLORIS Projected')

# SOWFA - FLORIS
ax = axarr[1,1]
wfct.visualization.visualize_cut_plane(
     cut_plane_difference, ax=ax, minSpeed=-1, maxSpeed=1)
ax.set_title('SOWFA - FLORIS Projected')

# # Plot the SOWFA flow and turbines using the input information
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8.5))

# hor_plane = si.get_hor_plane(90)
# wfct.visualization.visualize_cut_plane(
#     hor_plane, ax=ax2, minSpeed=minspeed, maxSpeed=maxspeed)
# vis.plot_turbines(ax2, si.layout_x, si.layout_y, si.yaw_angles, si.D)
# ax2.set_title('SOWFA')
# ax2.set_ylabel('y location [m]')


# # floris_flow_data_orig = fi.get_flow_data()

# # Plot the original FLORIS flow and turbines using the input information
# hor_plane_orig = fi.get_hor_plane(90)
# wfct.visualization.visualize_cut_plane(
#     hor_plane_orig, ax=ax1, minSpeed=minspeed, maxSpeed=maxspeed)
# vis.plot_turbines(ax1, fi.layout_x, fi.layout_y,
#                   fi.get_yaw_angles(),
#                   fi.floris.farm.turbine_map.turbines[0].rotor_diameter)
# ax1.set_title('FLORIS - Original')
# ax1.set_ylabel('y location [m]')

# # Set the relevant FLORIS parameters to equal the SOWFA case
# fi.reinitialize_flow_field(wind_speed=[si.precursor_wind_speed],
#                            wind_direction=[si.precursor_wind_dir],
#                            layout_array=(si.layout_x, si.layout_y)
#                            )

# # Set the yaw angles
# fi.calculate_wake(yaw_angles=si.yaw_angles)

# # Plot the FLORIS flow and turbines using the input information, and bounds from SOWFA
# hor_plane_matched = fi.get_hor_plane(90,x_bounds=[np.min(sowfa_flow_data.x), np.max(sowfa_flow_data.x)],
#                                         y_bounds=[np.min(sowfa_flow_data.y), np.max(sowfa_flow_data.y)])
# wfct.visualization.visualize_cut_plane(
#     hor_plane_matched, ax=ax3, minSpeed=minspeed, maxSpeed=maxspeed)
# vis.plot_turbines(ax3, fi.layout_x, fi.layout_y,
#                   fi.get_yaw_angles(),
#                   fi.floris.farm.turbine_map.turbines[0].rotor_diameter)
# ax3.set_title('FLORIS - Matched')
# ax3.set_xlabel('x location [m]')
# ax3.set_ylabel('y location [m]')

plt.show()
