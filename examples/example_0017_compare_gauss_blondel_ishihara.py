# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
import pandas as pd

# Define a quick function for getting arbitrary points from sowfa
from sklearn import neighbors

def get_points_from_flow_data(x_points,y_points,z_points,flow_data):
    X = np.column_stack([flow_data.x,flow_data.y,flow_data.z])
    n_neighbors = 1
    knn = neighbors.KNeighborsRegressor(n_neighbors)
    y_ = knn.fit(X, flow_data.u)

    # Predict new points
    T = np.column_stack([x_points,y_points,z_points])
    return knn.predict(T)

# Load the SOWFA case in
si = wfct.sowfa_utilities.SowfaInterface('sowfa_example')

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("example_input.json")

# Get HH and D
HH = fi.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi.floris.farm.turbines[0].rotor_diameter

wind_speed_mod = 0.3

# Match SOWFA
fi.reinitialize_flow_field(wind_speed=[si.precursor_wind_speed - wind_speed_mod],
                           wind_direction=[si.precursor_wind_dir],
                           layout_array=(si.layout_x, si.layout_y)
                           )

# Calculate wake
fi.calculate_wake(yaw_angles=si.yaw_angles)

# Repeat for Blondel
fi_b = wfct.floris_interface.FlorisInterface("example_input.json")
fi_b.floris.farm.set_wake_model('blondel')

fi_b.reinitialize_flow_field(
                        wind_speed=[si.precursor_wind_speed - wind_speed_mod],
                        wind_direction=[si.precursor_wind_dir],
                        layout_array=(si.layout_x, si.layout_y)
                        )

fi_b.calculate_wake(yaw_angles=si.yaw_angles)      

# Repeat for Ishihara-Qian
fi_iq = wfct.floris_interface.FlorisInterface("example_input.json")
fi_iq.floris.farm.set_wake_model('ishihara')

fi_iq.reinitialize_flow_field(
                        wind_speed=[si.precursor_wind_speed - wind_speed_mod],
                        wind_direction=[si.precursor_wind_dir],
                        layout_array=(si.layout_x, si.layout_y)
                        )

fi_iq.calculate_wake(yaw_angles=si.yaw_angles)  

# Set up points
step_size = 5
x_0 = si.layout_x[0]
y_0 = si.layout_y[0]
y_points = np.arange(-100+y_0,100+step_size+y_0,step_size)
x_points = np.ones_like(y_points) * 3 * D + x_0
z_points = np.ones_like(x_points) * HH

#  Make plot
fig, axarr = plt.subplots(5,2,figsize=(15,10),sharex='col',sharey='col')

for d_idx, d_downstream in enumerate([0,2,4,6,8]):

    # Grab x points
    x_points = np.ones_like(y_points) * d_downstream * D + x_0

    # Get the values
    flow_points = fi.get_set_of_points(x_points,y_points,z_points)
    flow_points_b = fi_b.get_set_of_points(x_points,y_points,z_points)
    flow_points_iq = fi_iq.get_set_of_points(x_points,y_points,z_points)
    sowfa_u = get_points_from_flow_data(x_points,y_points,z_points,si.flow_data)

    # Get horizontal plane at default height (hub-height)
    hor_plane = fi_b.get_hor_plane()

    ax = axarr[d_idx,0]
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    ax.plot(x_points,y_points,'r',lw=3)
    ax.set_title('%d D dowstream' % d_downstream)

    ax = axarr[d_idx,1]
    ax.plot(flow_points.y,flow_points.u,label='Gauss')
    ax.plot(flow_points_b.y,flow_points_b.u,label='Blondel')
    ax.plot(flow_points_iq.y,flow_points_iq.u,label='Ishihara-Qian')
    ax.plot(y_points,sowfa_u,label='SOWFA',color='k')
    ax.set_title('%d D dowstream' % d_downstream)
    ax.legend()
    ax.set_ylim([3,8])

# Center line plot
step_size = 5 
x_points = np.arange(0,x_0+D*12,step_size)
y_points = y_0+ np.zeros_like(x_points)
z_points = np.ones_like(x_points) * HH

# Get the values
flow_points = fi.get_set_of_points(x_points,y_points,z_points)
flow_points_b = fi_b.get_set_of_points(x_points,y_points,z_points)
flow_points_iq = fi_iq.get_set_of_points(x_points,y_points,z_points)
sowfa_u = get_points_from_flow_data(x_points,y_points,z_points,si.flow_data)

fig, ax = plt.subplots()
ax.plot((flow_points.x-x_0)/D,flow_points.u,label='Gauss')
ax.plot((flow_points.x-x_0)/D,flow_points_b.u,label='Blondel')
ax.plot((flow_points.x-x_0)/D,flow_points_iq.u,label='Ishihara-Qian')
ax.plot((x_points-x_0)/D,sowfa_u,label='SOWFA',color='k')
ax.set_title('Wake Centerline')
ax.legend()

print('SOWFA turbine powers: ', si.get_average_powers())
print('Gauss turbine powers: ', fi.get_turbine_power())
print('Blondel turbine powers: ', fi_b.get_turbine_power())
print('Ishihara-Qian turbine powers: ', fi_iq.get_turbine_power())

print('Gauss turbine avg ws: ', fi.floris.farm.turbines[0].average_velocity)
print('Blondel turbine avg ws: ', fi_b.floris.farm.turbines[0].average_velocity)
print('Ishihara-Qian turbine avg ws: ',
      fi_iq.floris.farm.turbines[0].average_velocity)

plt.show()