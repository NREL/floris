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
Example 01
Longitudinal

This script confirms the sameness of the basic gaussian models
"""

# Define some helper functions
def power_cross_sweep(fi_in,D,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations =  np.arange(1,10,0.5)
    power_out = np.zeros_like(sweep_locations)

    for x_idx, x_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,D*x_loc],[0,0]))
        fi.calculate_wake([yaw_angle,0])
        power_out[x_idx] = fi.get_turbine_power()[1]/1000.

    return sweep_locations, power_out

def power_cross_sweep_gain(fi_in,D,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(1,10,0.5)
    power_out = np.zeros_like(sweep_locations)

    for x_idx, x_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,D*x_loc],[0,0]))
        fi.calculate_wake([0,0])
        base_power = fi.get_turbine_power()[1]/1000.
        fi.calculate_wake([yaw_angle,0])
        power_out[x_idx] = 100 * (fi.get_turbine_power()[1]/1000. - base_power) / base_power

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

# Gauss_Legacy Class with GCH disabled and deflection multiplier = 1.2
fi_gm = wfct.floris_interface.FlorisInterface("../other_jsons/input_merge.json")
fi_dict['gm'] = fi_gm
color_dict['gm'] = 'go--'
label_dict['gm'] = 'gauss_blondel_merge'


# Set up a saved gauss 
saved_gauss = dict()
saved_gauss[(10,0)] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([182.07981963, 217.05252788, 249.78285665, 279.89395112,
       307.30801439, 333.81260862, 359.56938903, 384.6862233 ,
       424.48570247, 488.53762279, 548.24645338, 602.6843714 ,
       654.05796691, 702.60972415, 747.23412443, 789.23768748,
       828.92916199, 866.45556672]) ]
saved_gauss[(10,20)] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([ 314.43412242,  359.50841134,  407.76577206,  458.15193224,
        511.6212569 ,  566.2087768 ,  621.638757  ,  678.39928531,
        744.55430824,  818.32540855,  882.95555809,  939.02583365,
        987.8593348 , 1031.11394432, 1069.61158343, 1104.04369079,
       1134.99130155, 1162.5736868 ]) ]
saved_gauss[(6,0)] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([182.07981963, 217.05252788, 249.78285665, 279.89395112,
       307.30801439, 333.81260862, 359.56938903, 384.6862233 ,
       424.48570247, 488.53762279, 548.24645338, 602.6843714 ,
       654.05796691, 702.60972415, 747.23412443, 789.23768748,
       828.92916199, 866.45556672]) ]
saved_gauss[(6,20)] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([ 314.43412242,  359.50841134,  407.76577206,  458.15193224,
        511.6212569 ,  566.2087768 ,  621.638757  ,  678.39928531,
        744.55430824,  818.32540855,  882.95555809,  939.02583365,
        987.8593348 , 1031.11394432, 1069.61158343, 1104.04369079,
       1134.99130155, 1162.5736868 ]) ]
saved_gauss[(3,0)] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([182.07981963, 217.05252788, 249.78285665, 279.89395112,
       307.30801439, 333.81260862, 359.56938903, 384.6862233 ,
       424.48570247, 488.53762279, 548.24645338, 602.6843714 ,
       654.05796691, 702.60972415, 747.23412443, 789.23768748,
       828.92916199, 866.45556672]) ]
saved_gauss[(3,20)] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([ 314.43412242,  359.50841134,  407.76577206,  458.15193224,
        511.6212569 ,  566.2087768 ,  621.638757  ,  678.39928531,
        744.55430824,  818.32540855,  882.95555809,  939.02583365,
        987.8593348 , 1031.11394432, 1069.61158343, 1104.04369079,
       1134.99130155, 1162.5736868 ]) ]
saved_gauss[(10,"gain")] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([72.69026466, 65.63198542, 63.24810178, 63.68768614, 66.48484027,
       69.61875081, 72.88422651, 76.35133369, 75.40150443, 67.50509487,
       61.05084723, 55.80723148, 51.03544101, 46.75486389, 43.14276456,
       39.88735058, 36.92259286, 34.17579983]) ]
saved_gauss[(6,"gain")] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([72.69026466, 65.63198542, 63.24810178, 63.68768614, 66.48484027,
       69.61875081, 72.88422651, 76.35133369, 75.40150443, 67.50509487,
       61.05084723, 55.80723148, 51.03544101, 46.75486389, 43.14276456,
       39.88735058, 36.92259286, 34.17579983]) ]
saved_gauss[(3,"gain")] = [np.array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
       7.5, 8. , 8.5, 9. , 9.5]) ,np.array([72.69026466, 65.63198542, 63.24810178, 63.68768614, 66.48484027,
       69.61875081, 72.88422651, 76.35133369, 75.40150443, 67.50509487,
       61.05084723, 55.80723148, 51.03544101, 46.75486389, 43.14276456,
       39.88735058, 36.92259286, 34.17579983]) ]

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
            sweep_locations, ps = power_cross_sweep(fi_dict[fi_key],D,yaw)
            ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([100,1500])
            # Save for after clean up
            # if fi_key == 'g':
            #     print('saved_gauss[(%d,%d)] = [np.' % (dist_downstream,yaw), repr(sweep_locations),',np.',repr(ps),']')

# Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D downstream, Gain' % (dist_downstream))

    # First plot the saved results
    sweep_locations, ps = saved_gauss[(dist_downstream, 'gain')]
    ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

    for fi_key in fi_dict.keys():
        sweep_locations, ps = power_cross_sweep_gain(fi_dict[fi_key],D,yaw_angle=20)
        ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
        ax.set_ylim([0,100])
        # Save for after clean-up
        # if fi_key == 'g':
        #     print('saved_gauss[(%d,"gain")] = [np.' % (dist_downstream), repr(sweep_locations),',np.',repr(ps),']')

axarr[0,0].legend()
axarr[-1,0].set_xlabel('Downstream Distance (D)')
axarr[-1,1].set_xlabel('Downstream Distance (D)')
axarr[-1,2].set_xlabel('Downstream Distance (D)')
plt.show()

