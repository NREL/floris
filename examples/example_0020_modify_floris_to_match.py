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
from floris import Vec3
import matplotlib.pyplot as plt
import numpy as np

# Define a minspeed and maxspeed to use across visualiztions
minspeed = 4.0
maxspeed = 8.5


# Load the SOWFA case in
si = wfct.sowfa_utilities.SowfaInterface('sowfa_example')

# Plot the SOWFA flow and turbines using the input information
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,8.5))
sowfa_flow_field = si.flow_field
hor_plane = wfct.cut_plane.HorPlane(si.flow_field, 90)
wfct.visualization.visualize_cut_plane(hor_plane,ax=ax2,minSpeed=minspeed,maxSpeed=maxspeed)
vis.plot_turbines(ax2, si.layout_x, si.layout_y, si.yaw_angles, si.D)
ax2.set_title('SOWFA')
ax2.set_ylabel('y location [m]')

# Load the FLORIS case in
fi = wfct.floris_utilities.FlorisInterface("example_input.json")
fi.calculate_wake()
floris_flow_field_orig = fi.get_flow_field()

# Plot the original FLORIS flow and turbines using the input information
hor_plane_orig = cp.HorPlane(floris_flow_field_orig, 90)
wfct.visualization.visualize_cut_plane(hor_plane_orig,ax=ax1,minSpeed=minspeed,maxSpeed=maxspeed)
vis.plot_turbines(ax1, fi.layout_x, fi.layout_y, fi.get_yaw_angles(), fi.floris.farm.turbine_map.turbines[0].rotor_diameter)
ax1.set_title('FLORIS - Original')
ax1.set_ylabel('y location [m]')

# Set the relevant FLORIS parameters to equal the SOWFA case
fi.reinitialize_flow_field(wind_speed=si.precursor_wind_speed,
                wind_direction=si.precursor_wind_dir,
                layout_array=(si.layout_x, si.layout_y)
                )
# fi.floris.farm.set_turbine_locations(si.layout_x, si.layout_y, calculate_wake=False)
# Set the yaw angles
fi.calculate_wake(yaw_angles=si.yaw_angles)

# Generate and get a flow from original FLORIS file
floris_flow_field_matched = fi.get_flow_field()

# Trim the flow to match SOWFA
sowfa_domain_limits = [[np.min(sowfa_flow_field.x), np.max(sowfa_flow_field.x)],
                       [np.min(sowfa_flow_field.y), np.max(sowfa_flow_field.y)], 
                       [np.min(sowfa_flow_field.z), np.max(sowfa_flow_field.z)]]
floris_flow_field_matched = floris_flow_field_matched.crop(floris_flow_field_matched, sowfa_domain_limits[0], sowfa_domain_limits[1], sowfa_domain_limits[2] )

# Plot the FLORIS flow and turbines using the input information
hor_plane_matched = cp.HorPlane(floris_flow_field_matched, 90)
wfct.visualization.visualize_cut_plane(hor_plane_matched,ax=ax3,minSpeed=minspeed,maxSpeed=maxspeed)
vis.plot_turbines(ax3, fi.layout_x, fi.layout_y, fi.get_yaw_angles(), fi.floris.farm.turbine_map.turbines[0].rotor_diameter)
ax3.set_title('FLORIS - Matched')
ax3.set_xlabel('x location [m]')
ax3.set_ylabel('y location [m]')

plt.show()