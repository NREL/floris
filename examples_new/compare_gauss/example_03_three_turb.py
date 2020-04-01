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

# For debugging
# np.seterr(all='raise')

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

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D,dist_downstream*D*2],[0,0,y_loc*D]))
        fi.calculate_wake([yaw_angle,0,0])
        power_out[y_idx] = fi.get_turbine_power()[2]/1000.

    return sweep_locations, power_out

def power_cross_sweep_gain(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D,dist_downstream*D*2],[0,0,y_loc*D]))
        fi.calculate_wake([0,0,0])
        base_power = fi.get_turbine_power()[2]/1000.
        fi.calculate_wake([yaw_angle,0,0])
        power_out[y_idx] = 100 * (fi.get_turbine_power()[2]/1000. - base_power) / base_power

    return sweep_locations, power_out

# Set up the models ....
# ======================
fi_dict = dict()
color_dict = dict()
label_dict = dict()

# Gauss Class -- Current Default
fi_g = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_dict['g'] = fi_g
color_dict['g'] = 'r^-'
label_dict['g'] = 'current_default'

# Gauss_Legacy Class with GCH disabled and deflection multiplier = 1.2
fi_gl = wfct.floris_interface.FlorisInterface("../other_jsons/input_legacy.json")
fi_dict['gl'] = fi_gl
color_dict['gl'] = 'bo--'
label_dict['gl'] = 'gauss_legacy'

# Set up a saved gauss 
saved_gauss = dict()
saved_gauss[(10,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1435.94370406, 1323.89503547, 1217.15795972, 1140.26658923,
       1111.78491916, 1140.26658923, 1217.15795972, 1323.89503547,
       1435.94370406]) ]
saved_gauss[(10,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1337.24318379, 1254.84555316, 1186.68781729, 1146.78010767,
       1148.51442743, 1197.65054508, 1286.72278821, 1395.9458244 ,
       1500.24350567]) ]
saved_gauss[(6,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1438.93360213, 1232.23167916, 1000.43763306,  816.19974529,
        746.3504836 ,  816.19974529, 1000.43763306, 1232.23167916,
       1438.93360213]) ]
saved_gauss[(6,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1226.76528778, 1044.48135074,  897.85075458,  816.61889812,
        827.16686762,  940.29903358, 1131.86539695, 1343.01397005,
       1514.42885058]) ]
saved_gauss[(3,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1505.19140631, 1199.59728334,  775.07370251,  399.30350983,
        247.97049643,  399.30350983,  775.07370251, 1199.59728334,
       1505.19140631]) ]
saved_gauss[(3,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1314.00611798,  931.18272859,  547.7518679 ,  319.60029603,
        322.93021495,  554.43381933,  934.88567606, 1311.855186  ,
       1558.47919504]) ]
saved_gauss[(10,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-6.87356475, -5.21563118, -2.5033844 ,  0.57122768,  3.30365232,
        5.03250349,  5.71534926,  5.44233395,  4.47787761]) ]
saved_gauss[(6,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-14.74483006, -15.23660945, -10.25420027,   0.0513542 ,
        10.82820817,  15.20452426,  13.13702719,   8.99037841,
         5.24661099]) ]
saved_gauss[(3,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-12.70172601, -22.37538868, -29.32906043, -19.96055928,
        30.22928921,  38.85022437,  20.61893895,   9.35796573,
         3.54026661]) ]


# Get HH and D
HH = fi_gl.floris.farm.flow_field.turbine_map.turbines[0].hub_height
D = fi_gl.floris.farm.turbines[0].rotor_diameter

# Make a plot of comparisons
fig, axarr = plt.subplots(3,3,sharex=True, sharey=False,figsize=(14,9))

# Do the absolutes
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    for y_idx, yaw in enumerate([0 , 20]):
        ax = axarr[d_idx, y_idx]
        ax.set_title('%d D spacing, yaw = %d' % (dist_downstream,yaw))

        # First plot the saved results
        sweep_locations, ps = saved_gauss[(dist_downstream, yaw)]
        ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

        for fi_key in fi_dict.keys():
            sweep_locations, ps = power_cross_sweep(fi_dict[fi_key],D,dist_downstream,yaw)
            ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([0,1800])

            # Save for after clean up
            # if fi_key == 'g':
            #     print('saved_gauss[(%d,%d)] = [np.' % (dist_downstream,yaw), repr(sweep_locations),',np.',repr(ps),']')

# Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D spacing, Gain' % (dist_downstream))

    # First plot the saved results
    sweep_locations, ps = saved_gauss[(dist_downstream, 'gain')]
    ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

    for fi_key in fi_dict.keys():
        sweep_locations, ps = power_cross_sweep_gain(fi_dict[fi_key],D,dist_downstream,yaw_angle=20)
        ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
        ax.set_ylim([-40,40])
        # Save for after clean-up
        # if fi_key == 'g':
        #     print('saved_gauss[(%d,"gain")] = [np.' % (dist_downstream), repr(sweep_locations),',np.',repr(ps),']')

axarr[0,0].legend()
axarr[-1,0].set_xlabel('Lateral Offset (D)')
axarr[-1,1].set_xlabel('Lateral Offset (D)')
axarr[-1,2].set_xlabel('Lateral Offset (D)')
plt.show()

