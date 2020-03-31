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
Example 02
Test behavior for run of 5 turbines

"""

# TI = 0.06

# Define some helper functions
def power_cross_sweep(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D,dist_downstream*D*2,dist_downstream*D*3,dist_downstream*D*4],[0,0,0,0,y_loc*D]))
        fi.calculate_wake([yaw_angle,0,0,0,0])
        power_out[y_idx] = fi.get_turbine_power()[4]/1000.

    return sweep_locations, power_out

def power_cross_sweep_gain(fi_in,D,dist_downstream,yaw_angle=0):
    fi = copy.deepcopy(fi_in)

    sweep_locations = np.arange(-2,2.25,0.25)
    sweep_locations = np.arange(-1,1.25,0.25)
    power_out = np.zeros_like(sweep_locations)

    for y_idx, y_loc in enumerate(sweep_locations):

        fi.reinitialize_flow_field(layout_array=([0,dist_downstream*D,dist_downstream*D*2,dist_downstream*D*3,dist_downstream*D*4],[0,0,0,0,y_loc*D]))
        fi.calculate_wake([0,0,0,0,0])
        base_power = fi.get_turbine_power()[4]/1000.
        fi.calculate_wake([yaw_angle,0,0,0,0])
        power_out[y_idx] = 100 * (fi.get_turbine_power()[4]/1000. - base_power) / base_power

    return sweep_locations, power_out

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
saved_gauss[(10,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1475.857046  , 1438.61314837, 1407.12361597, 1385.76945702,
       1378.24142295, 1385.76945702, 1407.12361597, 1438.61314837,
       1475.857046  ]) ]
saved_gauss[(10,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1460.87607432, 1426.11836183, 1396.89636511, 1378.10577621,
       1373.68315683, 1385.03544661, 1410.70852317, 1446.48128931,
       1487.27870735]) ]
saved_gauss[(6,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1386.33360627, 1297.56151952, 1216.46738785, 1159.26620127,
       1138.5417156 , 1159.26620127, 1216.46738785, 1297.56151952,
       1386.33360627]) ]
saved_gauss[(6,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1330.80440754, 1255.38388012, 1190.62796652, 1149.27972502,
       1142.70862247, 1175.47342487, 1242.16104566, 1328.69052857,
       1417.72060005]) ]
saved_gauss[(3,0)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1308.75179417, 1105.39829949,  891.46814402,  726.85407202,
        664.85992705,  726.85407202,  891.46814402, 1105.39829949,
       1308.75179417]) ]
saved_gauss[(3,20)] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([1157.83042919,  954.9184752 ,  792.60776463,  703.31776021,
        702.4195797 ,  794.11740686,  962.28277757, 1163.62487709,
       1349.63535527]) ]
saved_gauss[(10,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-1.01506929, 12.02821879, 11.51990855, 10.8243142 , 10.17994975,
        9.80928765,  9.83854542, 10.23366611, 10.8826738 ]) ]
saved_gauss[(6,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-4.00547159, 10.38469059, 10.77897463, 10.47859704,  9.84211287,
        9.53432126,  9.92784683, 10.95353561, 12.06408527]) ]
saved_gauss[(3,"gain")] = [np.array([-1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,  1.  ]) ,np.array([-11.53170262,  -0.11975635,   2.63354986,   8.76905788,
        13.47688416,  13.97260825,  13.17746328,  12.82450192,
        13.00086758]) ]


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
        # ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

        for fi_key in fi_dict.keys():
            sweep_locations, ps = power_cross_sweep(fi_dict[fi_key],D,dist_downstream,yaw)
            ax.plot(sweep_locations,ps,color_dict[fi_key] ,label=label_dict[fi_key])
            ax.set_ylim([500,1600])

            # Save for after clean up
            # if fi_key == 'g':
            #     print('saved_gauss[(%d,%d)] = [np.' % (dist_downstream,yaw), repr(sweep_locations),',np.',repr(ps),']')

# Check the gains
for d_idx, dist_downstream in enumerate([10, 6, 3]):
    ax = axarr[d_idx, -1]
    ax.set_title('%d D spacing, Gain' % (dist_downstream))

    # First plot the saved results
    sweep_locations, ps = saved_gauss[(dist_downstream, 'gain')]
    # ax.plot(sweep_locations,ps,'ks-' ,label='Saved FLORIS Gauss',lw=3)

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

