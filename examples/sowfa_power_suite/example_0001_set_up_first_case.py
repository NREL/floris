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
import os
import pandas as pd


# Parameters
ti = 'hi'  #hi/low

# Load in SOWFA flow data and power results
df_flow = pd.read_pickle('flow_data_%s.p' % ti)
df_power = pd.read_pickle('data_sowfa.p')

if ti == 'hi':
    df_power = df_power[df_power.TI > 0.07]
    het_wind_speed_gain = 1.03
    hom_wind_speed = 8.25
else:
    df_power = df_power[df_power.TI < 0.07]
    het_wind_speed_gain = 1.02
    hom_wind_speed = 8.33

df_power = df_power[df_power.yaw_0==0]

# Iniitialize FLORIS
fi = wfct.floris_interface.FlorisInterface("example_input.json")

# Pick a random case and set FLORIS to match
random_case = df_power.sample()
print(random_case)
x_locations = np.array(random_case.layout_x.values[0])
y_locations = np.array(random_case.layout_y.values[0])
fi.reinitialize_flow_field(wind_speed=[hom_wind_speed],layout_array =[x_locations,y_locations])

yaw_array = np.array([random_case.yaw_0.values,random_case.yaw_1.values,random_case.yaw_2.values,random_case.yaw_3.values])
sowfa_power_array = np.array([random_case.sowfa_power_0.values,random_case.sowfa_power_1.values,random_case.sowfa_power_2.values,random_case.sowfa_power_3.values])
fi.calculate_wake(yaw_angles=yaw_array)

# Get horizontal plane at default height (hub-height)


# Plot and show homogenous flow
fig, axarr = plt.subplots(3,2,sharex='row',sharey='row',figsize=(8,10))

ax = axarr[0,0]
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Homogenous')

# Compare homogenous results
floris_power_array = np.array([p[0]/1000. for p in fi.get_turbine_power()])

het_wind_speed = het_wind_speed_gain * df_flow.u.values

ax = axarr[1,0]
row_locs = [1,3]
ax.plot(x_locations[row_locs],sowfa_power_array[row_locs],'.-',color='k',label='SOWFA')
ax.plot(x_locations[row_locs],floris_power_array[row_locs],'.-',color='r',label='FLORIS')
ax.set_title('Top Row')

ax = axarr[2,0]
row_locs = [0,2]
ax.plot(x_locations[row_locs],sowfa_power_array[row_locs],'.-',color='k',label='SOWFA')
ax.plot(x_locations[row_locs],floris_power_array[row_locs],'.-',color='r',label='FLORIS')
ax.set_title('Bottom Row')

# Redo to Het

fi.reinitialize_flow_field( wind_speed=het_wind_speed.tolist(),#wind_direction=270. * np.ones_like(df_flow.u.values),
                                wind_layout=[df_flow.x.values, df_flow.y.values])

fi.calculate_wake()

# Compare homogenous results
floris_power_array = np.array([p[0]/1000. for p in fi.get_turbine_power()])

ax = axarr[0,1]
hor_plane = fi.get_hor_plane()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Heterogenous')

ax = axarr[1,1]
row_locs = [1,3]
ax.plot(x_locations[row_locs],sowfa_power_array[row_locs],'.-',color='k',label='SOWFA')
ax.plot(x_locations[row_locs],floris_power_array[row_locs],'.-',color='r',label='FLORIS')
ax.set_title('Top Row')

ax = axarr[2,1]
row_locs = [0,2]
ax.plot(x_locations[row_locs],sowfa_power_array[row_locs],'.-',color='k',label='SOWFA')
ax.plot(x_locations[row_locs],floris_power_array[row_locs],'.-',color='r',label='FLORIS')
ax.set_title('Bottom Row')

plt.show()

