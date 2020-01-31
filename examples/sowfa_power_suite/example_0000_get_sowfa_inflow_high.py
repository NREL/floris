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

from floris.utilities import Vec3
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# # Define a minspeed and maxspeed to use across visualiztions
minspeed = 4.0
maxspeed = 8.5

for ti in ['low','hi']:

    # Load the SOWFA case in
    sowfa_root = '/Users/pfleming/Box Sync/sowfa_library/full_runs/near_wake'
    inflow_case = '%s_no_turbine' % ti

    si = wfct.sowfa_utilities.SowfaInterface(os.path.join(sowfa_root,inflow_case))

    flow_origin = si.flow_data.origin
    print("origin of saved flow field = ", flow_origin)

    # Re-origin flow to 0
    si.flow_data.x = si.flow_data.x + flow_origin.x1
    si.flow_data.y = si.flow_data.y + flow_origin.x2
    si.flow_data.origin = Vec3(0,0,0)


    # Get the hub-height velocities at turbine locations
    y_points = np.arange(200,1600,10.) 
    x_points = np.ones_like(y_points) * 1000. 
    z_points = np.ones_like(y_points) * 90.

    u_points = si.flow_data.get_points_from_flow_data(x_points,y_points,z_points)

    x_points_2 = np.ones_like(y_points) * 2000.
    u_points_2 = si.flow_data.get_points_from_flow_data(x_points_2,y_points,z_points)

    x_points_3 = np.ones_like(y_points) * 3000. 
    u_points_3 = si.flow_data.get_points_from_flow_data(x_points_3,y_points,z_points)

    # Grab a set of points to describe the flow field by
    x_p = np.array([950,950,950,950,1500,1500,1500,1500,2000,2000,2000,2000])
    y_p = np.array([300,700,1100,1500,300,700,1100,1500,300,700,1100,1500]) 
    z_p = z_points = np.ones_like(x_p) * 90.
    u_p = si.flow_data.get_points_from_flow_data(x_p,y_p,z_p)

    # Save the flow points as a dataframe
    df = pd.DataFrame({'x':x_p,
                        'y':y_p,
                        'z':z_p,
                        'u':u_p})
    df.to_pickle('flow_data_%s.p' % ti)

    # Plot the SOWFA flow and turbines using the input information
    fig, axarr = plt.subplots(2, 1, figsize=(5, 5))

    ax = axarr[0]
    sowfa_flow_data = si.flow_data
    hor_plane = si.get_hor_plane(90)
    wfct.visualization.visualize_cut_plane(
        hor_plane, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    ax.plot(x_points,y_points,color='k')
    ax.plot(x_points_2,y_points,color='g')
    ax.plot(x_points_3,y_points,color='r')
    ax.scatter(x_p,y_p,color='m')

    ax.set_title('SOWFA')
    ax.set_ylabel('y location [m]')

    ax = axarr[1]
    ax.plot(y_points,u_points,color='k')
    ax.plot(y_points,u_points_2,color='g')
    ax.plot(y_points,u_points_3,color='r')
    vis.reverse_cut_plane_x_axis_in_plot(ax)
    ax.set_title('Looking downstream')
    fig.suptitle(ti)




plt.show()
