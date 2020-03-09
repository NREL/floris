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

"""
Example 00
First in set of scripts to test the behaviors of new gaussian models

This script confirms the sameness of the basic gaussian models
"""

# Define some helper functions
def power_cross_sweep(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D],[0,y_loc*D]))
        fi.calculate_wake([yaw_angle,0])
        power_out[y_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out

def power_cross_sweep_gain(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D],[0,y_loc*D]))
        fi.calculate_wake([0,0])
        base_power = fi.get_turbine_power()[1]/1000.
        fi.calculate_wake([yaw_angle,0])
        power_out[y_idx] = 100 * (fi.get_turbine_power()[1]/1000. - base_power) / base_power

    return sweep_locations, power_out

# Set up the models ....
# ======================
fi_dict = dict()
color_dict = dict()
label_dict = dict()

# Gauss Class (This one is going away I think?)
fi_g = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_g.floris.farm.set_wake_model('gauss')
fi_dict['g'] = fi_g
color_dict['g'] = 'ks-'
label_dict['g'] = 'gauss'

# Gauss Legacy Class
fi_gl = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_gl.floris.farm.set_wake_model('legacy_gauss')
fi_dict['gl'] = fi_gl
color_dict['gl'] = 'ro--'
label_dict['gl'] = 'legacy_gauss'

# Gauss Merge Class
fi_gm = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_gm.floris.farm.set_wake_model('merge_gauss')
fi_dict['gm'] = fi_gm
color_dict['gm'] = 'b^-'
label_dict['gm'] = 'merge_gauss'

# Get HH and D
HH = fi_gl.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi_gl.floris.farm.turbines[0].rotor_diameter

# Make a plot of comparisons
fig, axarr = plt.subplots(3,3,sharex=True, sharey=False,figsize=(14,9))

# Do the absolutes
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    for y_idx, yaw in enumerate([0 , 20]):
        ax = axarr[d_idx, y_idx]
        ax.set_title('%d D downstream, yaw = %d' % (dist_downstream,yaw))
        for fi_key in fi_dict.keys():
            sweep_locations, ps = power_cross_sweep(fi_dict[fi_key],D,dist_downstream,yaw)
            ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([0,2000])

# Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D downstream, Gain' % (dist_downstream))
    for fi_key in fi_dict.keys():
        sweep_locations, ps = power_cross_sweep_gain(fi_dict[fi_key],D,dist_downstream,yaw_angle=20)
        ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
        ax.set_ylim([-100,100])

axarr[0,0].legend()
axarr[-1,0].set_xlabel('Lateral Offset (D)')
axarr[-1,1].set_xlabel('Lateral Offset (D)')
axarr[-1,2].set_xlabel('Lateral Offset (D)')
plt.show()


# # Downstream power comparison D
# d_dowstream = 7
# # sweep_locations, ps_fi_g = power_cross_sweep(fi_g,D,d_dowstream)
# sweep_locations, ps_fi_gmg = power_cross_sweep(fi_gmg,D,d_dowstream)
# sweep_locations, ps_fi_gmb = power_cross_sweep(fi_gmb,D,d_dowstream)
# sweep_locations, ps_fi_gmgm = power_cross_sweep(fi_gmgm,D,d_dowstream)
# ax = axarr[0,2]
# ax.set_title('Crosstream power sweep at %dD' % d_dowstream)
# #ax.plot(sweep_locations,ps_fi_g,label='Gauss',lw=2,color=color_dict['Gauss'])
# #ax.plot(sweep_locations,ps_fi_b,label='Blondel',lw=2,color=color_dict['Blondel'])
# ax.plot(sweep_locations,ps_fi_gmg,'--',label='gmg',color=color_dict['gmg'])
# ax.plot(sweep_locations,ps_fi_gmb,'-',label='gmb',color=color_dict['gmb'],lw=3)
# ax.plot(sweep_locations,ps_fi_gmgm,'.-',label='gmgm',color=color_dict['gmgm'])
# ax.set_ylabel('Downstream Power (kW)')
# ax.set_xlabel('Lateral Offset (D)')
# ax.grid(True)
# ax.legend()