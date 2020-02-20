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

This example demonstrates the new gauss-merge class which currently includes 3 possibilities, 
a gauss legacy model, a direct implementation of blondel model, and a new hybrid designed
to ensure consistency with legacy gauss starting around 5~6D

Gauss - as in develop (not needed once confirmed a match to GM-G)
Blondel - as in develop (not needed once confirmed a match to GM-B)
GM-G - gauss in gauss-merge, but functionally unchanged
GM-B - blondel in gauss- merge, but functionally unchanged
GM-GM - new version of blondel
"""

# PARAMETERS
# Can run against low or hi TI sowfa simulations
ti = 'hi' #'low' or 'hi'

# First design a set of functions which demonstrate agreement

# Define some helper functions
def power_sweep(fi_in,D):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(1,10,0.25)
    power_out = np.zeros_like(sweep_locations)

    for x_idx, x_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,x_loc*D],[0,0]))
        fi.calculate_wake()
        power_out[x_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out

# Define some helper functions
def power_cross_sweep(fi_in,D,dist_downstream):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
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

# # Center line c value
# def cl_Cx(fi_in, D, HH):
#     fi = copy.deepcopy(fi_in)
#     fi.reinitialize_flow_field(layout_array=([0],[0]))
#     sweep_locations = np.arange(1,10,0.25)

#     x_points = sweep_locations * D
#     y_points = np.zeros_like(x_points)
#     z_points = np.ones_like(x_points) * HH
#     flow_points = fi.get_set_of_points_temp_hack(x_points,y_points,z_points)
#     return sweep_locations, flow_points.Cx

# Color palatte
color = sns.color_palette("hls", 5)
color_dict = dict()
color_dict['Gauss'] = color[0]
color_dict['Blondel'] = color[1]
color_dict['gmg'] = color[2]
color_dict['gmb'] = color[3]
color_dict['gmgm'] = color[4]

# Load in SOWFA results
df_power_full = pd.read_pickle('data_sowfa.p')


if ti == 'hi':
    df_power_ti = df_power_full[df_power_full.TI > 0.07]
    ti_val = 0.09
    wind_speed = 8.25
else:
    df_power_ti = df_power_full[df_power_full.TI < 0.07]
    ti_val = 0.065
    wind_speed = 8.33


# Initialize the FLORIS interface fi
fi_g = wfct.floris_interface.FlorisInterface("example_input.json")
fi_g.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val])

# Get HH and D
HH = fi_g.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi_g.floris.farm.turbines[0].rotor_diameter

# Repeat for Blondel
fi_b = wfct.floris_interface.FlorisInterface("example_input.json")
fi_b.floris.farm.set_wake_model('blondel')
fi_b.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val])

# Repeat for Gauss-M-Gauss
fi_gmg = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gmg.floris.farm.set_wake_model('gauss_m')
fi_gmg.floris.farm.wake.velocity_model.model_code = 'g'
fi_gmg.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val])

# Repeat for Gauss-M-Blondel
fi_gmb = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gmb.floris.farm.set_wake_model('gauss_m')
fi_gmb.floris.farm.wake.velocity_model.model_code = 'b'
fi_gmb.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val])

# Repeat for Gauss-M-Blondel
fi_gmgm = wfct.floris_interface.FlorisInterface("example_input.json")
fi_gmgm.floris.farm.set_wake_model('gauss_m')
fi_gmgm.floris.farm.wake.velocity_model.model_code = 'gm'
fi_gmgm.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[ti_val])

# Add a d_spacing column to SOWFA
df_power_ti['d_spacing'] = [(lx[2] - lx[0])/D for lx in df_power_ti.layout_x.values]

# force yaw = 0
df_power_ti = df_power_ti[df_power_ti.yaw_0==0]


# Make the comparison figure
fig, axarr = plt.subplots(3,3,sharex=False,sharey=False,figsize=(15,10))

# Downstream power comparison
sweep_locations, ps_fi_g = power_sweep(fi_g,D)
sweep_locations, ps_fi_b = power_sweep(fi_b,D)
sweep_locations, ps_fi_gmg = power_sweep(fi_gmg,D)
sweep_locations, ps_fi_gmb = power_sweep(fi_gmb,D)
sweep_locations, ps_fi_gmgm = power_sweep(fi_gmgm,D)
ax = axarr[0,0]
ax.set_title('Dowstream power sweep')
# ax.plot(sweep_locations,ps_fi_g,label='Gauss',lw=2,color=color_dict['Gauss'])
# ax.plot(sweep_locations,ps_fi_b,label='Blondel',lw=2,color=color_dict['Blondel'])
ax.plot(df_power_ti.d_spacing,df_power_ti.sowfa_power_2, color='k',marker='o',ls='None',label='SOWFA 1')
ax.plot(df_power_ti.d_spacing,df_power_ti.sowfa_power_3, color='k',marker='s',ls='None',label='SOWFA 2')
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('Downstream Power (kW)')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()

# Downstream power comparison D
d_dowstream = 2
# sweep_locations, ps_fi_g = power_cross_sweep(fi_g,D,d_dowstream)
# sweep_locations, ps_fi_b = power_cross_sweep(fi_b,D,d_dowstream)
sweep_locations, ps_fi_gmg = power_cross_sweep(fi_gmg,D,d_dowstream)
sweep_locations, ps_fi_gmb = power_cross_sweep(fi_gmb,D,d_dowstream)
sweep_locations, ps_fi_gmgm = power_cross_sweep(fi_gmgm,D,d_dowstream)
ax = axarr[0,1]
ax.set_title('Crosstream power sweep at %dD' % d_dowstream)
#ax.plot(sweep_locations,ps_fi_g,label='Gauss',lw=2,color=color_dict['Gauss'])
#ax.plot(sweep_locations,ps_fi_b,label='Blondel',lw=2,color=color_dict['Blondel'])
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('Downstream Power (kW)')
ax.set_xlabel('Lateral Offset (D)')
ax.grid(True)
ax.legend()

# Downstream power comparison D
d_dowstream = 7
# sweep_locations, ps_fi_g = power_cross_sweep(fi_g,D,d_dowstream)
# sweep_locations, ps_fi_b = power_cross_sweep(fi_b,D,d_dowstream)
sweep_locations, ps_fi_gmg = power_cross_sweep(fi_gmg,D,d_dowstream)
sweep_locations, ps_fi_gmb = power_cross_sweep(fi_gmb,D,d_dowstream)
sweep_locations, ps_fi_gmgm = power_cross_sweep(fi_gmgm,D,d_dowstream)
ax = axarr[0,2]
ax.set_title('Crosstream power sweep at %dD' % d_dowstream)
#ax.plot(sweep_locations,ps_fi_g,label='Gauss',lw=2,color=color_dict['Gauss'])
#ax.plot(sweep_locations,ps_fi_b,label='Blondel',lw=2,color=color_dict['Blondel'])
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('Downstream Power (kW)')
ax.set_xlabel('Lateral Offset (D)')
ax.grid(True)
ax.legend()

# Center line velocity comparison
sweep_locations, ps_fi_gmg = cl_vel(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_vel(fi_gmb,D,HH)
sweep_locations, ps_fi_gmgm = cl_vel(fi_gmgm,D,HH)
ax = axarr[1,0]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('Center line velocity (m/s)')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()


# Center line sigma comparison
sweep_locations, ps_fi_gmg = cl_sigma(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_sigma(fi_gmb,D,HH)
sweep_locations, ps_fi_gmgm = cl_sigma(fi_gmgm,D,HH)
ax = axarr[1,1]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('Center line sigma')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()

# Center line n comparison
sweep_locations, ps_fi_gmg = cl_n(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_n(fi_gmb,D,HH)
sweep_locations, ps_fi_gmgm = cl_n(fi_gmgm,D,HH)
ax = axarr[2,0]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('N Value')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()

# Center line beta comparison
sweep_locations, ps_fi_gmg = cl_beta(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_beta(fi_gmb,D,HH)
sweep_locations, ps_fi_gmgm = cl_beta(fi_gmgm,D,HH)
ax = axarr[2,1]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
ax.set_ylabel('Beta Value')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()


# Center line beta comparison
sweep_locations, ps_fi_gmg = cl_C(fi_gmg,D,HH)
sweep_locations, ps_fi_gmb = cl_C(fi_gmb,D,HH)
sweep_locations, ps_fi_gmgm = cl_C(fi_gmgm,D,HH)
# sweep_locations, ps_fi_gmgmx = cl_Cx(fi_gmgm,D,HH)
ax = axarr[2,2]
ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
# ax.plot(sweep_locations,ps_fi_gmb2x,'.-',label='Cx',color='k')
ax.set_ylabel('C Value')
ax.set_xlabel('Distance downstream (D)')
ax.grid(True)
ax.legend()



plt.show()

