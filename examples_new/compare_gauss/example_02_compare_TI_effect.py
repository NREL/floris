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
def ti_sweep(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    ti_values = np.arange(0.04,0.15,0.005)
    power_out = np.zeros_like(ti_values)

    for ti_idx, ti in enumerate(ti_values):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D],[0,0]),turbulence_intensity=[ti])
        fi.calculate_wake([yaw_angle,0])
        power_out[ti_idx] = fi.get_turbine_power()[1]/1000.

    return ti_values, power_out

def ti_sweep_gain(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)


    ti_values = np.arange(0.04,0.15,0.005)
    power_out = np.zeros_like(ti_values)

    for ti_idx, ti in enumerate(ti_values):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D],[0,0]),turbulence_intensity=[ti])
        fi.calculate_wake([0,0])
        base_power = fi.get_turbine_power()[1]/1000.
        fi.calculate_wake([yaw_angle,0])
        power_out[ti_idx] = 100 * (fi.get_turbine_power()[1]/1000. - base_power) / base_power

    return ti_values, power_out

# Set up the models ....
# ======================
fi_dict = dict()
color_dict = dict()
label_dict = dict()

# Gauss Class -- Current Default
fi_g = wfct.floris_interface.FlorisInterface("../example_input.json")
# fi_g.floris.farm.set_wake_model('gauss')
# fi_g.set_gch(True)
fi_dict['g'] = fi_g
color_dict['g'] = 'r^-'
label_dict['g'] = 'gauss'

# Gauss_Legacy Class with GCH disabled and deflection multiplier = 1.2
fi_gl = wfct.floris_interface.FlorisInterface("../other_jsons/example_input_legacy.json")
# fi_gl.floris.farm.set_wake_model('gauss_legacy')
# fi_gl.set_gch(False) # Disable GCH
# fi_gl.floris.farm.wake._deflection_model.deflection_multiplier = 1.2 # Deflection multiplier to 1.2
fi_dict['gl'] = fi_gl
color_dict['gl'] = 'bo--'
label_dict['gl'] = 'gauss_legacy'


# Set up a saved gauss 
saved_gauss = dict()
saved_gauss[(10,0)] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([ 710.37278464,  761.44551604,  810.51984522,  857.39042285,
        901.95198686,  943.2742517 ,  982.07342148, 1018.61804967,
       1052.99339377, 1085.29854836, 1115.64037967, 1144.10367431,
       1170.35963506, 1194.99856921, 1218.12599809, 1239.84265436,
       1260.24395932, 1279.41976946, 1297.45431555, 1314.42627671,
       1330.40894657, 1345.47045975]) ]
saved_gauss[(10,20)] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([1177.8352177 , 1176.39581191, 1177.56634904, 1181.3229298 ,
       1187.46445136, 1195.68934213, 1205.65217418, 1217.00241491,
       1229.40877991, 1242.57282345, 1256.23500202, 1270.17581067,
       1284.21393222, 1298.20276282, 1312.02622225, 1325.59442034,
       1338.83951516, 1351.71194062, 1364.17707935, 1376.21239517,
       1387.80500348, 1398.94964023]) ]
saved_gauss[(6,0)] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([ 395.38103375,  428.21405187,  468.39667203,  508.72400682,
        548.24645338,  586.20892283,  623.5166273 ,  660.04729796,
        695.70857729,  729.71457636,  762.35531004,  793.9617257 ,
        824.5225837 ,  854.03674333,  882.51118069,  909.95933632,
        935.71010296,  960.47836556,  984.29274196, 1007.18273903,
       1029.17908025, 1050.31320429]) ]
saved_gauss[(6,20)] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([ 814.63964374,  830.21807143,  848.73912014,  866.16946293,
        882.95555809,  899.42411477,  915.64810699,  931.63541565,
        947.72683929,  963.94718095,  980.28874066,  996.72239724,
       1013.20608054, 1029.69104521, 1046.12636623, 1062.4620403 ,
       1078.65102213, 1094.65046669, 1110.42239317, 1125.93393909,
       1141.15733282, 1155.82716289]) ]
saved_gauss[(3,0)] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([272.16015991, 281.11187872, 289.94778828, 298.67718779,
       307.30801439, 315.84709402, 324.30033663, 332.67289018,
       340.96926343, 349.19342494, 357.34888368, 365.4387552 ,
       373.46581637, 381.43255105, 389.25622806, 396.7104533 ,
       414.70052122, 438.84461948, 462.67638667, 486.19674545,
       509.40411088, 532.29547069]) ]
saved_gauss[(3,20)] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([483.05378444, 490.3244278 , 497.505388  , 504.60258843,
       511.6212569 , 518.5660268 , 525.44102068, 532.24991967,
       538.77701758, 545.16910465, 551.50284522, 557.78068125,
       564.00482878, 570.17730378, 576.29994446, 582.37443071,
       601.94516033, 621.3572529 , 640.35201582, 658.96855715,
       677.23787838, 695.18443439]) ]
saved_gauss[(10,"gain")] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([65.80522835, 54.49507379, 45.28531978, 37.78121359, 31.65495156,
       26.7594594 , 22.76599161, 19.47583447, 16.75370303, 14.49133746,
       12.60214536, 11.01929302,  9.72814627,  8.63634453,  7.70858058,
        6.91634262,  6.23653502,  5.65038722,  5.14259061,  4.70061498,
        4.31416649,  3.97475694]) ]
saved_gauss[(6,"gain")] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([106.03913041,  93.87922181,  81.20092879,  70.26313902,
        61.05084723,  53.43064217,  46.85223567,  41.14676608,
        36.22468807,  32.09920868,  28.58685809,  25.53783954,
        22.88396953,  20.56753451,  18.53972948,  16.75928779,
        15.27619705,  13.96929967,  12.81424172,  11.79043241,
        10.88034675,  10.04595183]) ]
saved_gauss[(3,"gain")] = [np.array([0.04 , 0.045, 0.05 , 0.055, 0.06 , 0.065, 0.07 , 0.075, 0.08 ,
       0.085, 0.09 , 0.095, 0.1  , 0.105, 0.11 , 0.115, 0.12 , 0.125,
       0.13 , 0.135, 0.14 , 0.145]) ,np.array([77.48879358, 74.42323321, 71.58447421, 68.94580807, 66.48484027,
       64.18261767, 62.02296493, 59.99197271, 58.01336817, 56.12238539,
       54.33176663, 52.63314942, 51.01913055, 49.48312676, 48.05156679,
       46.80087854, 45.15177327, 41.58935198, 38.40170674, 35.53536985,
       32.94707756, 30.60123046]) ]


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

        # # First plot the saved results
        ti_values, ps = saved_gauss[(dist_downstream, yaw)]
        ax.plot(ti_values,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

        for fi_key in fi_dict.keys():
            ti_values, ps = ti_sweep(fi_dict[fi_key],D,dist_downstream,yaw)
            ax.plot(ti_values,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([0,2000])
            # Save for after clean up
            if fi_key == 'g':
                print('saved_gauss[(%d,%d)] = [np.' % (dist_downstream,yaw), repr(ti_values),',np.',repr(ps),']')
# print('GAINS')
# # Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D downstream, Gain' % (dist_downstream))

    # First plot the saved results
    ti_values, ps = saved_gauss[(dist_downstream, 'gain')]
    ax.plot(ti_values,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

    for fi_key in fi_dict.keys():
        ti_values, ps = ti_sweep_gain(fi_dict[fi_key],D,dist_downstream,yaw_angle=20)
        ax.plot(ti_values,ps,color_dict[fi_key] ,label=label_dict[fi_key])
        ax.set_ylim([-100,100])
        # Save for after clean-up
        if fi_key == 'g':
            print('saved_gauss[(%d,"gain")] = [np.' % (dist_downstream), repr(ti_values),',np.',repr(ps),']')

axarr[0,0].legend()
axarr[-1,0].set_xlabel('TI')
axarr[-1,1].set_xlabel('TI')
axarr[-1,2].set_xlabel('TI')
plt.show()

