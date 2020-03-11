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

# Gauss Class -- Current Default
fi_g = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_g.floris.farm.set_wake_model('gauss')
# fi_g.set_gch(True)
fi_dict['g'] = fi_g
color_dict['g'] = 'r^-'
label_dict['g'] = 'gauss'

# Gauss_Legacy Class with GCH disabled and deflection multiplier = 1.2
fi_gl = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_gl.floris.farm.set_wake_model('gauss_legacy')
fi_gl.set_gch(False) # Disable GCH
fi_gl.floris.farm.wake._deflection_model.deflection_multiplier = 1.2 # Deflection multiplier to 1.2
fi_dict['gl'] = fi_gl
color_dict['gl'] = 'bo--'
label_dict['gl'] = 'gauss_legacy'

# Set up a saved gauss 
saved_gauss = dict()
saved_gauss[(10,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1533.60441684, 1365.62088487, 1155.08543545,  973.56022679,
        901.95198686,  973.56022679, 1155.08543545, 1365.62088487,
       1533.60441684]) ]
saved_gauss[(10,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1240.69409842, 1054.31942699,  964.51369145, 1018.28384293,
       1187.46445136, 1389.68299861, 1549.58603932, 1639.79197365,
       1676.52144747]) ]
saved_gauss[(6,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1582.68316317, 1366.9223727 , 1026.53731597,  692.43879565,
        548.24645338,  692.43879565, 1026.53731597, 1366.9223727 ,
       1582.68316317]) ]
saved_gauss[(6,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1331.00738281,  987.67770715,  695.83692699,  647.76067736,
        882.95555809, 1231.17777609, 1510.96750781, 1645.69394558,
       1683.41250921]) ]
saved_gauss[(3,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1610.71175613, 1373.78219575,  949.99041557,  509.82923117,
        307.30801439,  509.82923117,  949.99041557, 1373.78219575,
       1610.71175613]) ]
saved_gauss[(3,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1531.15831746, 1193.3884595 ,  745.50596379,  428.65487254,
        511.6212569 ,  911.48845717, 1339.98284702, 1599.96793975,
       1678.00383106]) ]
saved_gauss[(10,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-19.09947019, -22.79559879, -16.4984977 ,   4.5938212 ,
        31.65495156,  42.74237591,  34.15337011,  20.07666197,
         9.31902837]) ]
saved_gauss[(6,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-15.90184228, -27.74441864, -32.21513566,  -6.4522841 ,
        61.05084723,  77.80311904,  47.19070455,  20.39410419,
         6.3644669 ]) ]
saved_gauss[(3,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([ -4.9390239 , -13.13117442, -21.5248963 , -15.9218722 ,
        66.48484027,  78.78309078,  41.05224906,  16.46445446,
         4.17778505]) ]




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

        # First plot the saved results
        sweep_locations, ps = saved_gauss[(dist_downstream, yaw)]
        ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

        for fi_key in fi_dict.keys():
            sweep_locations, ps = power_cross_sweep(fi_dict[fi_key],D,dist_downstream,yaw)
            ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([0,2000])
            # Save for after clean up
            # if fi_key == 'g':
            #     print('saved_gauss[(%d,%d)] = [np.' % (dist_downstream,yaw), repr(sweep_locations),',np.',repr(ps),']')
# print('GAINS')
# Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D downstream, Gain' % (dist_downstream))

    # First plot the saved results
    sweep_locations, ps = saved_gauss[(dist_downstream, 'gain')]
    ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

    for fi_key in fi_dict.keys():
        sweep_locations, ps = power_cross_sweep_gain(fi_dict[fi_key],D,dist_downstream,yaw_angle=20)
        ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
        ax.set_ylim([-100,100])
        # Save for after clean-up
        # if fi_key == 'g':
        #     print('saved_gauss[(%d,"gain")] = [np.' % (dist_downstream), repr(sweep_locations),',np.',repr(ps),']')

axarr[0,0].legend()
axarr[-1,0].set_xlabel('Lateral Offset (D)')
axarr[-1,1].set_xlabel('Lateral Offset (D)')
axarr[-1,2].set_xlabel('Lateral Offset (D)')
plt.show()

