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
import copy
import seaborn as sns

# Legend
"""
Gauss - as in develop
Blondel - as in develop
GM-G - gauss in gauss-merge, but functionally unchanged
GM-G2 - new version of gauss
GM-B - blondel in gauss- merge, but functionally unchanged
GM-B2 - new version of blondel
"""


# Define some helper functions
def power_sweep(fi_in,D):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(1,10,0.25)
    power_out = np.zeros_like(sweep_locations)

    for x_idx, x_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,x_loc*D],[0,0]))
        fi.calculate_wake()
        # print(fi.get_turbine_power())
        power_out[x_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out

# Define some helper functions
def power_cross_sweep(fi_in,D):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    dist_downstream = 7
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D],[0,y_loc*D]))
        fi.calculate_wake()
        # print(fi.get_turbine_power())
        power_out[y_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out

# Center line velocity
def cl_vel(fi_in, D, HH):
    fi = copy.deepcopy(fi_in)
    fi.reinitialize_flow_field(layout_array=([0],[0]))
    sweep_locations = np.arange(1,10,0.25)

    x_points = sweep_locations * D
    y_points = np.zeros_like(x_points)
    z_points = np.ones_like(x_points) * HH
    flow_points = fi.get_set_of_points(x_points,y_points,z_points)
    return sweep_locations, flow_points.u

# Center line sigma
def cl_sigma(fi_in, D, HH):
    fi = copy.deepcopy(fi_in)
    fi.reinitialize_flow_field(layout_array=([0],[0]))
    sweep_locations = np.arange(1,10,0.25)

    x_points = sweep_locations * D
    y_points = np.zeros_like(x_points)
    z_points = np.ones_like(x_points) * HH
    flow_points = fi.get_set_of_points_temp_hack(x_points,y_points,z_points)
    return sweep_locations, flow_points.sigma_tilde

# Center line n value
def cl_n(fi_in, D, HH):
    fi = copy.deepcopy(fi_in)
    fi.reinitialize_flow_field(layout_array=([0],[0]))
    sweep_locations = np.arange(1,10,0.25)

    x_points = sweep_locations * D
    y_points = np.zeros_like(x_points)
    z_points = np.ones_like(x_points) * HH
    flow_points = fi.get_set_of_points_temp_hack(x_points,y_points,z_points)
    return sweep_locations, flow_points.n

# Center line beta value
def cl_beta(fi_in, D, HH):
    fi = copy.deepcopy(fi_in)
    fi.reinitialize_flow_field(layout_array=([0],[0]))
    sweep_locations = np.arange(1,10,0.25)

    x_points = sweep_locations * D
    y_points = np.zeros_like(x_points)
    z_points = np.ones_like(x_points) * HH
    flow_points = fi.get_set_of_points_temp_hack(x_points,y_points,z_points)
    return sweep_locations, flow_points.beta

# Center line c value
def cl_C(fi_in, D, HH):
    fi = copy.deepcopy(fi_in)
    fi.reinitialize_flow_field(layout_array=([0],[0]))
    sweep_locations = np.arange(1,10,0.25)

    x_points = sweep_locations * D
    y_points = np.zeros_like(x_points)
    z_points = np.ones_like(x_points) * HH
    flow_points = fi.get_set_of_points_temp_hack(x_points,y_points,z_points)
    return sweep_locations, flow_points.C

# Center line c value
def cl_Cx(fi_in, D, HH):
    fi = copy.deepcopy(fi_in)
    fi.reinitialize_flow_field(layout_array=([0],[0]))
    sweep_locations = np.arange(1,10,0.25)

    x_points = sweep_locations * D
    y_points = np.zeros_like(x_points)
    z_points = np.ones_like(x_points) * HH
    flow_points = fi.get_set_of_points_temp_hack(x_points,y_points,z_points)
    return sweep_locations, flow_points.Cx

# Color palatte
color = sns.color_palette("hls", 5)
color_dict = dict()
color_dict['Gauss'] = color[0]
color_dict['Blondel'] = color[1]
color_dict['gmg'] = color[2]
color_dict['gmb'] = color[3]
color_dict['gmb2'] = color[4]


# Initialize the FLORIS interface fi
fi_g = wfct.floris_interface.FlorisInterface("example_input.json")

# Get HH and D
HH = fi_g.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi_g.floris.farm.turbines[0].rotor_diameter


# Repeat for Blondel
fi_b = wfct.floris_interface.FlorisInterface("example_input.json")
fi_b.floris.farm.set_wake_model('blondel')

# Repeat for Gauss-M-Gauss
fi_gmg = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gmg.floris.farm.set_wake_model('gauss_m')
fi_gmg.floris.farm.wake.velocity_model.model_code = 'g'

# Repeat for Gauss-M-Blondel
fi_gmb = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gmb.floris.farm.set_wake_model('gauss_m')
fi_gmb.floris.farm.wake.velocity_model.model_code = 'b'

# Repeat for Gauss-M-Blondel
fi_gmb2 = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gmb2.floris.farm.set_wake_model('gauss_m')
fi_gmb2.floris.farm.wake.velocity_model.model_code = 'b2'





# Make the comparison figure
fig, axarr = plt.subplots(3,3,sharex=False,sharey=False,figsize=(10,11))

# Downstream power comparison
sweep_locations, ps_fi_g = power_sweep(fi_g,D)
sweep_locations, ps_fi_b = power_sweep(fi_b,D)
sweep_locations, ps_fi_gmg = power_sweep(fi_gmg,D)
sweep_locations, ps_fi_gmb = power_sweep(fi_gmb,D)
sweep_locations, ps_fi_gmb2 = power_sweep(fi_gmb2,D)
ax = axarr[0,0]
ax.set_title('Dowstream power sweep')
# ax.plot(sweep_locations,ps_fi_g,label='Gauss',lw=2,color=color_dict['Gauss'])
# ax.plot(sweep_locations,ps_fi_b,label='Blondel',lw=2,color=color_dict['Blondel'])
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.set_ylabel('Downstream Power (kW)')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()

# Downstream power comparison
sweep_locations, ps_fi_g = power_cross_sweep(fi_g,D)
sweep_locations, ps_fi_b = power_cross_sweep(fi_b,D)
sweep_locations, ps_fi_gmg = power_cross_sweep(fi_gmg,D)
sweep_locations, ps_fi_gmb = power_cross_sweep(fi_gmb,D)
sweep_locations, ps_fi_gmb2 = power_cross_sweep(fi_gmb2,D)
ax = axarr[0,1]
ax.set_title('Crosstream power sweep at 7D')
#ax.plot(sweep_locations,ps_fi_g,label='Gauss',lw=2,color=color_dict['Gauss'])
#ax.plot(sweep_locations,ps_fi_b,label='Blondel',lw=2,color=color_dict['Blondel'])

ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.set_ylabel('Downstream Power (kW)')
ax.set_xlabel('Lateral Offset (D)')
ax.grid(True)
ax.legend()

# Center line velocity comparison
sweep_locations, ps_fi_gmg = cl_vel(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_vel(fi_gmb,D,HH)
sweep_locations, ps_fi_gmb2 = cl_vel(fi_gmb2,D,HH)
ax = axarr[1,0]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.set_ylabel('Center line velocity (m/s)')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()


# Center line sigma comparison
sweep_locations, ps_fi_gmg = cl_sigma(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_sigma(fi_gmb,D,HH)
sweep_locations, ps_fi_gmb2 = cl_sigma(fi_gmb2,D,HH)
ax = axarr[1,1]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.set_ylabel('Center line sigma')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()

# Center line n comparison
sweep_locations, ps_fi_gmg = cl_n(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_n(fi_gmb,D,HH)
sweep_locations, ps_fi_gmb2 = cl_n(fi_gmb2,D,HH)
ax = axarr[2,0]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.set_ylabel('N Value')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()

# Center line beta comparison
sweep_locations, ps_fi_gmg = cl_beta(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_beta(fi_gmb,D,HH)
sweep_locations, ps_fi_gmb2 = cl_beta(fi_gmb2,D,HH)
ax = axarr[2,1]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.set_ylabel('Beta Value')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()


# Center line beta comparison
sweep_locations, ps_fi_gmg = cl_C(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_C(fi_gmb,D,HH)
sweep_locations, ps_fi_gmb2 = cl_C(fi_gmb2,D,HH)
sweep_locations, ps_fi_gmb2x = cl_Cx(fi_gmb2,D,HH)
ax = axarr[2,2]
# ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmb2,'.-',label='gmb2',color=color_dict['gmb2'])
ax.plot(sweep_locations,ps_fi_gmb2x,'.-',label='Cx',color='k')
ax.set_ylabel('C Value')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()



plt.show()


# wind_speed_mod = 0.3

# # Match SOWFA
# fi.reinitialize_flow_field(wind_speed=[si.precursor_wind_speed - wind_speed_mod],
#                            wind_direction=[si.precursor_wind_dir],
#                            layout_array=(si.layout_x, si.layout_y)
#                            )

# # Calculate wake
# fi.calculate_wake(yaw_angles=si.yaw_angles)

# # Repeat for Blondel
# fi_b = wfct.floris_interface.FlorisInterface("example_input.json")
# fi_b.floris.farm.set_wake_model('blondel')

# fi_b.reinitialize_flow_field(
#                         wind_speed=[si.precursor_wind_speed - wind_speed_mod],
#                         wind_direction=[si.precursor_wind_dir],
#                         layout_array=(si.layout_x, si.layout_y)
#                         )

# fi_b.calculate_wake(yaw_angles=si.yaw_angles)      

# # Repeat for Ishihara-Qian
# fi_iq = wfct.floris_interface.FlorisInterface("example_input.json")
# fi_iq.floris.farm.set_wake_model('ishihara')

# fi_iq.reinitialize_flow_field(
#                         wind_speed=[si.precursor_wind_speed - wind_speed_mod],
#                         wind_direction=[si.precursor_wind_dir],
#                         layout_array=(si.layout_x, si.layout_y)
#                         )

# fi_iq.calculate_wake(yaw_angles=si.yaw_angles)  

# # Set up points
# step_size = 5
# x_0 = si.layout_x[0]
# y_0 = si.layout_y[0]
# y_points = np.arange(-100+y_0,100+step_size+y_0,step_size)
# x_points = np.ones_like(y_points) * 3 * D + x_0
# z_points = np.ones_like(x_points) * HH

# #  Make plot
# fig, axarr = plt.subplots(5,2,figsize=(15,10),sharex='col',sharey='col')

# for d_idx, d_downstream in enumerate([0,2,4,6,8]):

#     # Grab x points
#     x_points = np.ones_like(y_points) * d_downstream * D + x_0

#     # Get the values
#     flow_points = fi.get_set_of_points(x_points,y_points,z_points)
#     flow_points_b = fi_b.get_set_of_points(x_points,y_points,z_points)
#     flow_points_iq = fi_iq.get_set_of_points(x_points,y_points,z_points)
#     sowfa_u = get_points_from_flow_data(x_points,y_points,z_points,si.flow_data)

#     # Get horizontal plane at default height (hub-height)
#     hor_plane = fi_b.get_hor_plane()

#     ax = axarr[d_idx,0]
#     wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
#     ax.plot(x_points,y_points,'r',lw=3)
#     ax.set_title('%d D dowstream' % d_downstream)

#     ax = axarr[d_idx,1]
#     ax.plot(flow_points.y,flow_points.u,label='Gauss')
#     ax.plot(flow_points_b.y,flow_points_b.u,label='Blondel')
#     ax.plot(flow_points_iq.y,flow_points_iq.u,label='Ishihara-Qian')
#     ax.plot(y_points,sowfa_u,label='SOWFA',color='k')
#     ax.set_title('%d D dowstream' % d_downstream)
#     ax.legend()
#     ax.set_ylim([3,8])

# # Center line plot
# step_size = 5 
# x_points = np.arange(0,x_0+D*12,step_size)
# y_points = y_0+ np.zeros_like(x_points)
# z_points = np.ones_like(x_points) * HH

# # Get the values
# flow_points = fi.get_set_of_points(x_points,y_points,z_points)
# flow_points_b = fi_b.get_set_of_points(x_points,y_points,z_points)
# flow_points_iq = fi_iq.get_set_of_points(x_points,y_points,z_points)
# sowfa_u = get_points_from_flow_data(x_points,y_points,z_points,si.flow_data)

# fig, ax = plt.subplots()
# ax.plot((flow_points.x-x_0)/D,flow_points.u,label='Gauss')
# ax.plot((flow_points.x-x_0)/D,flow_points_b.u,label='Blondel')
# ax.plot((flow_points.x-x_0)/D,flow_points_iq.u,label='Ishihara-Qian')
# ax.plot((x_points-x_0)/D,sowfa_u,label='SOWFA',color='k')
# ax.set_title('Wake Centerline')
# ax.legend()

# print('SOWFA turbine powers: ', si.get_average_powers())
# print('Gauss turbine powers: ', fi.get_turbine_power())
# print('Blondel turbine powers: ', fi_b.get_turbine_power())
# print('Ishihara-Qian turbine powers: ', fi_iq.get_turbine_power())

# print('Gauss turbine avg ws: ', fi.floris.farm.turbines[0].average_velocity)
# print('Blondel turbine avg ws: ', fi_b.floris.farm.turbines[0].average_velocity)
# print('Ishihara-Qian turbine avg ws: ',
#       fi_iq.floris.farm.turbines[0].average_velocity)

# plt.show()